/*
 *  ATRV Driver and Ros Node by John Folkesson modified from
 *  Atrv Driver
 *  David Lu!! - 2/2010
 *  Modified from Player driver
 *
 *  Player - One Hell of a Robot Server
 *  Copyright (C) 2000
 *     Brian Gerkey, Kasper Stoy, Richard Vaughan, & Andrew Howard
 *
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
#include <atrv/atrv.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <termios.h>
#include <sys/stat.h>
#include <iostream>
#include <unistd.h>
#include <string.h> // memcpy
#include <fcntl.h>
#include <math.h>
//finds the sign of a value
static long sgn( long val ) {
    if (val < 0)
        return 0;
    else
        return 1;
}

static unsigned int getInt16( const unsigned char *bytes ) {
    unsigned int i;
    memcpy( &i, bytes, 2 );
    return(htons(i));
}


static unsigned long getInt32( const unsigned char *bytes ) {
    unsigned long i;
    memcpy( &i, bytes, 4 );
    return(htonl(i));
}


static void putInt8( unsigned int i, unsigned char *bytes ) {
    memcpy( bytes, &i, 1 );
}

static void putInt32( unsigned long l, unsigned char *bytes ) {
    uint32_t conv;
    conv = htonl( l );
    memcpy( bytes, &conv, 4 );
}

ATRV::ATRV() {
  herp = 0;
  found = false;
  offset = 0;
  odomReady = 0;
  last_distance=0;
  last_bearing=0;
  newSonar=false;
  useSonar=false;
  useDIO=false;
  updateTimer = 99;
  //cofigurations params
  configure("sonarMaxRange", SONAR_MAX_RANGE);
  configure("sonarRangeConversion", RANGE_CONVERSION);
  configure("odoDistanceConversion",(long)ODO_DISTANCE_CONVERSION);
  configure("odoAngleConversion",ODO_ANGLE_CONVERSION);
  configure("sonarEchoDelay",SONAR_ECHO_DELAY);
  configure("sonarPingDelay",SONAR_PING_DELAY);
  configure("sonarSetDelay",SONAR_SET_DELAY);
  //TODO find perfect period. It seems sensitive.
  period=25000; //Bigger is slower
  //Was 25000
}
ATRV::~ATRV() {
  serialOn=0;
  pthread_mutex_destroy(&writeMutex);

  pthread_join(thread,NULL);

}
void ATRV::configure(std::string name, long val){
  if (name.compare("odoDistanceConversion")==0){
      odoDistanceConversion=val;
      return;
  }else if (name.compare("odoAngleConversion")==0){
      odoAngleConversion=val;
      return;
  }else if (name.compare("sonarEchoDelay")==0){
      sonarEchoDelay=val;
      return;
  }else if (name.compare("sonarPingDelay")==0){
       sonarPingDelay=val;
      return;
  }else  if (name.compare("sonarSetDelay")==0){
      sonarSetDelay=val;
      return;
  }else if (name.compare("sonarMaxRange")==0){
      sonarMaxRange=val;
      return;
  }else if (name.compare("sonarRangeConversion")==0){
      sonarRangeConversion=val;
      return;
  }
}



int ATRV::initialize(const char* device_name) {
    // Open the port
    ROS_INFO("in initialize");
    fd = open(device_name, O_RDWR | O_NONBLOCK);
    if (fd == -1) {
        fprintf(stderr,"Could not open serial port %s\n", device_name );
        return -1;
    }

    // Get the terminal info
    struct termios info;
    if (tcgetattr(fd, &info) < 0) {
        fprintf(stderr,"Could not get terminal information for %s\n", device_name );
        return -1;
    }

    // Turn off echo, canonical mode, extended processing, signals, break signal, cr to newline, parity off, 8 bit strip, flow control,
    // size, parity bit, and output processing
    // info.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG | BRKINT | ICRNL | INPCK | ISTRIP | IXON | CSIZE | PARENB | OPOST);

    // Set size to 8 bits
    // info.c_cflag |= CS8;

    // Miguel Additions
    info.c_iflag &= ~(BRKINT | PARMRK | INPCK | ISTRIP | INLCR | IGNCR | ICRNL | IXON | IXOFF | IUCLC | IMAXBEL);
    info.c_lflag &= ~(ISIG | ICANON | IEXTEN | ECHO | ECHOE | ECHOK | ECHONL | NOFLSH | XCASE | TOSTOP | FLUSHO);
    info.c_oflag &= ~(OPOST | OCRNL | ONLCR | ONOCR | OFILL | OFDEL);
    info.c_cflag &= ~(PARENB | PARODD | HUPCL | CSTOPB | CRTSCTS);

    info.c_iflag |= (IGNBRK | IGNPAR);
    // info.c_lflag |= ();
    // info.c_oflag |= ( | );
    info.c_cflag |= (CS8 | CLOCAL);

    // Set time and bytes to enable read at once
    info.c_cc[VTIME] = 0;
    info.c_cc[VMIN] = 0;
    speed_t baud_rate = B115200;
    // speed_t baud_rate = B38400;
    if (cfsetospeed(&info, baud_rate) < 0) {
        fprintf(stderr,"Could not set the output speed for %s\n", device_name );
        return -1;
    }

    if (cfsetispeed(&info, baud_rate) < 0) {
        fprintf(stderr,"Could not set the input speed for %s\n", device_name );
        return -1;
    }

    // Actually set the controls on the terminal
    if (tcsetattr(fd, TCSAFLUSH, &info) < 0) {
        close(fd);
        fprintf(stderr,"Could not set controls on serial port %s\n", device_name );
    }
    pthread_mutex_init(&writeMutex, NULL);
    parserState.init();
    serialOn=1;
    pthread_create(&thread, NULL, ATRV::readThread, this);

    setOdometryPeriod (period);
    if (useDIO)
      setDigitalIoPeriod(period);
    motionSetDefaults();
    //setSonarPower(true);
    return 0;
}



//____________________________________________________________________
//__________________________READ FROM ATRV__________________________
//____________________________________________________________________

void* ATRV::readThread(void *ptr) {
  ATRV *atrv = static_cast<ATRV *>(ptr);
  // Set up the read set to include the serial port
  //Erik's comment: was inside the loop before
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(atrv->fd, &read_set);
  while ((atrv->fd>=0)&&(atrv->serialOn)) {





    // Is there any new data to be read from the atrv?
    if (select(atrv->fd + 1, &read_set, NULL, NULL, NULL) < 0) {
    }
    else if (FD_ISSET(atrv->fd, &read_set)) {
      atrv->readPacket();
    }
  }
  return NULL;
}
void ATRV::readPacket() {
  parserState.packetTime=ros::Time::now();
  // If there's no packet ready, just return
  const int read_size = readData();
  if (read_size == 0)
    return;
  // Check to make sure that the packet is the correct size
  const int data_size = read_size - PROTOCOL_SIZE;
  if (readBuffer[PACKET_SIZE_BYTE] != data_size) {
    for (int i = 0; i < BUFFER_SIZE; ++i) {
    }
  }
  // Calculate the packet CRC and verify that it matches
  if (computeCRC(readBuffer + PACKET_CRC_START, data_size + PACKET_CRC_OFFSET) != readBuffer[data_size + PACKET_DATA_START_BYTE]) {
    for (int i = 0; i < BUFFER_SIZE; ++i) {
    }
    // Eat everything up to the end of the packet
    unsigned char tdata = 0;
    while (tdata != ETX)
      while (read(fd, &tdata, 1) != 1) { }

    return;
  }
  parsePacket(readBuffer);
}


int ATRV::readData() {
  // Read one byte of of the packet.  No need to check for errors, since this will be called repeatedly.
  int bRead = read(fd, readBuffer + offset, 1);
  if (bRead == 0)
    return 0;
  else if (bRead < 0) {
    printf("Error reading from port!\n");
    return 0;
  }
  // Have we started a packet yet?
  if (!found) {
    // If the first character isn't an ESC, the packet is invalid.  Reset the offset and return.  This
    // will eat badly-formed packets.
    if (readBuffer[0] != ESC) {
      offset = 0;
      return 0;
    }
    if (offset == 0) {
      offset = 1;
      return 0;
    }
    // We have to wait for a STX to show up before it's a valid packet.  If we see an ESC, then we just
    // keep looking for an STX.  If we see something else, give up and start looking for a new packet.
    if (readBuffer[1] == STX) {
      found = true;
      offset = 2;
      return 0;
    } else if (readBuffer[1] == ESC) {
      offset = 1;
      return 0;
    } else {
      offset = 0;
      return 0;
    }
  } else {
    // If the previous character was an ESC,
    if (readBuffer[offset - 1] == ESC) {
      switch (readBuffer[offset]) {
      case NUL:  // Skip over NULs
	read(fd, readBuffer + offset, 1);
	// Should we be checking the return code here?
	++offset;
	return 0;
      case SOH:  // Ignore SOHs by deleting them
	--offset;
	return 0;
      case ETX: // ETX ends the packet, so return the length
	const int retval = offset + 1;
	found = false;
	offset = 0;
	return retval;
      };
    } else {
      // Just increment the counter
      ++offset;
      return 0;
    }
  }
  // Should never get here
  return 0;
}
//____________________________________________________________________
//__________________________PARSE FUNCTIONS__________________________
//____________________________________________________________________

void ATRV::parseMotReport( const unsigned char* buffer ) {
  int rv, timeStamp, acc, trq;
  unsigned char axis;
 if (buffer[PACKET_OPCODE_BYTE]== MOT_SYSTEM_REPORT) {
    rv        = getInt32(&(buffer[6]));
    timeStamp = getInt32(&(buffer[10]));

    axis = buffer[14];
    int ds=getInt32(&(buffer[15])); //SOMETIMES GIVES STRANGE VALUES THAT ARE CORRECTED BELOW

    if (axis == 0) {
      if (!odomReady&1) {
	last_distance=ds;
	odomReady = odomReady | 1;
      }
      parserState.odoTime=parserState.packetTime;
      double ddis=((double)ds-(double)last_distance)
	/(double)odoDistanceConversion;
      //DO NOTHING IF TOO GREAT A VALUE (THE CORRECTION MENTIONED ABOVE)
      if (abs(ddis)>1.0){
        acc = getInt32(&(buffer[23]));
          trq = getInt32(&(buffer[27]));
          int vel_temp = getInt32(&(buffer[19]));
          printf("here %d %d %d %d\n", (int)buffer[0],(int)buffer[1],(int)buffer[2],(int)buffer[3]);
          //printf("ddis=%d\tacc=%d\ttrq=%d\tvel_temp=%d\n",ddis,acc,trq,vel_temp);
          return;
      }
      last_distance=ds;
      parserState.x +=ddis*cos(parserState.theta);
      parserState.y +=ddis*sin(parserState.theta);
      parserState.v = ddis; //CHANGED THIS from the following:
      //parserState.v = (double)getInt32(&(buffer[19]))/(double)odoDistanceConversion; //FUNKAR INTE BRA :(

//      printf("x=%f\n",parserState.x);
  //    printf("y=%f\n",parserState.y);
    //  printf("theta=%f\n",parserState.theta);
      /*
      double test=(double)getInt32(&(buffer[19]));
      printf("v=%f\n",test);
       */
      //printf("oDC=%f\n", odoDistanceConversion);


           //printf("odo trans ");
       //parserState.print();
    } else if (axis == 1) {
      if (odomReady<2){
	last_bearing=ds;
	odomReady = odomReady | 2;
      }
      double ddis=((double)ds-(double)last_bearing)/(double)odoAngleConversion/2.0; //WORKS FINE, JUST HAD TO ADJUST THE VALUE WITH A FACTOR OF 0.5
      //printf("drot = %f\n",ddis);
      last_bearing=ds;
      parserState.theta+= ddis; //WORKS FINE RADIANS!!
      //printf("theta = %f\n",parserState.theta);

      parserState.omega = ddis; //CHANGED THIS from the following:
      //parserState.omega = (double)getInt32(&(buffer[19]))/(double)odoAngleConversion;

      parserState.odoTime=parserState.packetTime;
      parserState.readyToUpdate=1;
      //printf("odo rot ");
      //parserState.print();
    }
    acc = getInt32(&(buffer[23]));
    trq = getInt32(&(buffer[27]));
 }
}

//processes a digital io packet from the atrv
void ATRV::parseDioReport( const unsigned char* buffer ) {
  unsigned char length = buffer[5];
  if (length >= 7)
    if ((buffer[PACKET_OPCODE_BYTE]==DIO_REPORT)||
	(buffer[PACKET_OPCODE_BYTE]==DIO_UPDATE)) {
      unsigned long timeStamp = getInt32(&(buffer[6]));
      unsigned char address = buffer[10];  //NetID
      unsigned short data = getInt16(&(buffer[11])); //offset, State
      //if (address == HEADING_HOME_ADDRESS) {
      //home_bearing = parserState.theta;
      //}
      // check if the dio packet came from a bumper packet
      //else
      if (address == BUMPER_ADDRESS) {
	parserState.bump= data;
	if (data>0){
	  setBrakePower(false);
	  setMovement(0,0,1);
	}
      }
    }
}

//processes a sys packet from the atrv - and saves the data in the
//struct for later use, sys is primarily used for bat voltage & brake status
void ATRV::parseSysReport( const unsigned char* buffer ) {
  unsigned long timeStamp;
  unsigned char length = buffer[5];
  if (buffer[PACKET_OPCODE_BYTE]== SYS_STATUS){
    if (length < 9) {
      fprintf(stderr, "Got bad Sys packet (status)\n");
      return;
    }
    timeStamp=getInt32(&(buffer[6]));
    // raw voltage measurement...needs calibration offset added
    double v=getInt32(&(buffer[10]));
    if (v==0.0)
      parserState.voltage=0.0;
    else
      parserState.voltage=v/100.0 + POWER_OFFSET;
  }
  parserState.brake=buffer[14];
}



//parses a packet from the atrv, and decides what to do with it
void ATRV::parsePacket( const unsigned char* buffer ) {
  switch (buffer[PACKET_PORT_BYTE]) {
  case SYS_PORT:
    parseSysReport( buffer );
    break;
  case MOT_PORT:
    parseMotReport( buffer );
    break;
  case JSTK_PORT:
    //    parseJoyReport( buffer );
    break;
  case SONAR_PORT:
    parseSonarReport( buffer );
    break;
  case DIO_PORT:
    parseDioReport( buffer );
    break;
  case IR_PORT:

    break;
  default:
    break;
  }
  if (parserState.bump>0){
    parserState.bumpTimer=ros::Time::now();
  }
  //  setSonarUpdate();
  //  if (parserState.bump>0)
  //setDIOUpdate();

  if (parserState.readyToUpdate){
    //Comes in here at rate of period ot rflex MOTOR messages
    // right after parsing the odometry rot
    //odo rotation is sent directly after odometry translation.
    parserState.readyToUpdate=0;
    if (updateTimer*period>=50000) {
      //was 50000
      //this is a bit sensitive.
      //Too fast seems to overfill the embedded controller input buffer.
      //Too slow and the connection times out causing the robot to stop.
      //sendSystemStatusCommand();
      //printf("SSSSSSSMMMMMMMMMMMAAAAAAAAANNNNNNNGGGGG½!!!!!!\n");

      updateTimer = 0;
    }
    updateTimer++;

    /*
    double v,r,a;
    bool s;
    com.getAll(v,r,a,s);
    //we send motion commands as often as we get odometry
    //Erik's comment: is this really smart? we might overflow the buffer as stated before.
    setMovement(v, r, a);

    publish(s);

    */
  }
}

void ATRV::main_update(){ //Written by Erik Sillén
  double v,r,a;
    bool s;
    com.getAll(v,r,a,s);
     setMovement(v, r, a);
     publish(s);
}

//processes a sonar packet from the atrv
void ATRV::parseSonarReport( const unsigned char* buffer ) {
  int retval, timeStamp, count;
  unsigned char dlen = buffer[5];
  //dlen is either 0 8 or 26

  switch (buffer[PACKET_OPCODE_BYTE]) {
  case SONAR_REPORT:
    if (dlen>8){
      //printf("sonar0-3 %d %d %d %d\n", (int)buffer[0],(int)buffer[1],(int)buffer[2],(int)buffer[3]);
      //printf("sonar4-7 %d %d %d %d\n", (int)buffer[4],(int)buffer[5],(int)buffer[6],(int)buffer[7]);
      retval    = getInt32(&(buffer[6]));
      timeStamp = getInt32(&(buffer[10]));
      count = 0;
      while ((8+count*3<dlen) && (count<256) && (count < SONAR_MAX_COUNT)) {
	unsigned int sid = buffer[14+count*3];
	if (sid<12)
	  parserState.sonar[sid] = getInt16( &(buffer[14+count*3+1]) );
	//printf("sonar %d %d %d\n", sid, (int)buffer[14+count*3+1],(int)buffer[14+count*3]+2);
	count++;
      }
      newSonar=true;
      //printf("dlen %d\n", dlen);
      parserState.sonarTime=parserState.packetTime;
      parserState.print();
    }
    break;
  default:
    break;
  }
}

bool ATRV::sendCommand(const unsigned char port, const unsigned char id, const unsigned char opcode, const int length, unsigned char* data) {
  pthread_mutex_lock(&writeMutex);

  // Header
  writeBuffer[0] = ESC;
  writeBuffer[1] = STX;
  writeBuffer[PACKET_PORT_BYTE] = port;
  writeBuffer[PACKET_ID_BYTE] = id;
  writeBuffer[PACKET_OPCODE_BYTE] = opcode;
  writeBuffer[PACKET_SIZE_BYTE] = static_cast<unsigned char>(length);
  for (int i=0; i<length; i++) {
    writeBuffer[6+i] = data[i];
  }
  // Footer
  writeBuffer[length + PACKET_DATA_START_BYTE] = computeCRC(writeBuffer + PACKET_CRC_START, length + PACKET_CRC_OFFSET);
  writeBuffer[length + PACKET_DATA_START_BYTE + 1] = ESC;
  writeBuffer[length + PACKET_DATA_START_BYTE + 2] = ETX;
  //printf("write port %d %d %lf\n",port, opcode, parserState.packetTime.toSec());
  int ret = writePacket(length + 9);
  pthread_mutex_unlock(&writeMutex);

  return ret;
}


bool ATRV::writePacket(const int length) const {
  if (fd<0)
    return false;

  int bytes_written = 0;

  while (bytes_written < length) {
    int n = write(fd, writeBuffer + bytes_written, length - bytes_written);
    if (n < 0)
      return false;
    else
      bytes_written += n;

    // Put in a short wait to let the atrv controller catch up
    usleep(1000);
  }

  return true;
}

unsigned char ATRV::computeCRC(const unsigned char *buffer, const int n) {
  int crc =buffer[0];
  for (int i = 1; i < n; ++i)
    crc ^= buffer[i];
  return crc;
}
//_________________________________________________________________________________________________________________________________________
//_____________GET FUNCTIONS__________________________________________


bool ATRV::isPluggedIn()  {
  float v = parserState.voltage;
    if (v>PLUGGED_THRESHOLD)
        return true;
    else
        return false;
}

int ATRV::getNumSonars()  {
    return 12;
}

void ATRV::getSonarReadings(float* readings) {
  for (int j = 0;j<12;j++) {
    int range = parserState.sonar[j];
    if (range > sonarMaxRange)
      range = sonarMaxRange;
    float fRange = range /(double)sonarRangeConversion;
    readings[j] = fRange;
  }
}


//_________________________________________________________________________________________________________________________________________
//_____________SET/SEND FUNCTIONS__________________________________________

void ATRV::configureSonar(const unsigned long echo_delay, const unsigned long ping_delay,
                           const unsigned long set_delay, const unsigned long val) {

    unsigned char data[MAX_COMMAND_LENGTH];
    putInt32( echo_delay, &(data[0]) );
    putInt32( ping_delay, &(data[4]) );
    putInt32( set_delay , &(data[8]) );
    putInt8(  val, &(data[12]) );
    //sendCommand(SONAR_PORT, 4, SONAR_RUN, 13, data );
    sendCommand(SONAR_PORT, 4, SONAR_RUN, 13, data );
}
void ATRV::setSonarUpdate( ) {
    unsigned char data[MAX_COMMAND_LENGTH];
    putInt32( 1, &(data[0]) );
    sendCommand(SONAR_PORT, 4, SONAR_GET_UPDATE, 4, data );
}
void ATRV::setBrakePower( const bool on ) {
    int brak;
    if (on)
        brak = MOT_BRAKE_SET;
    else
        brak = MOT_BRAKE_RELEASE;
    sendCommand(MOT_PORT, 0, brak, 0, NULL );
}
void ATRV::setSonarPower(bool on) {
    unsigned long echo, ping, set, val;
    if (on) {
        echo = sonarEchoDelay;
        ping = sonarPingDelay;
        set = sonarSetDelay;
        val = 2;
    } else {
        echo = ping = set = val = 0;
    }
    configureSonar(echo, ping, set, val);
    com.setSonarPower(on);
 }


void ATRV::motionSetDefaults(  ) {
    sendCommand(MOT_PORT, 0, MOT_SET_DEFAULTS, 0, NULL );
}

void ATRV::setDigitalIoPeriod( const long period ) {
    unsigned char data[MAX_COMMAND_LENGTH];
    putInt32( period, &(data[0]) );
    sendCommand(DIO_PORT, 0, DIO_REPORTS_REQ, 4, data );
}
void ATRV::setDIOUpdate( ) {
    unsigned char data[MAX_COMMAND_LENGTH];
    putInt32( 1, &(data[0]) );
    sendCommand(DIO_PORT, 0, DIO_GET_UPDATE, 4, data );
}

void ATRV::setOdometryPeriod( const long period ) {
    unsigned char data[MAX_COMMAND_LENGTH];
    long mask;
    if (period==0)
        mask = 0;
    else
        mask = 3;
    putInt32( period, &(data[0]) );         /* period in micros */
    putInt32( mask, &(data[4]) );           /* mask  ie report both axis*/
    sendCommand(MOT_PORT, 0, MOT_SYSTEM_REPORT_REQ, 8, data );
}

void ATRV::setVelocity( const long tvel, const long rvel, const long acceleration) {
  ros::Time bumpt=parserState.bumpTimer;
  ros::Duration dur=ros::Time::now()-bumpt;
  double d=dur.toSec();
  long utvel =labs(tvel);
  long urvel =labs(rvel);
  long acc =acceleration;
  if (d<15){ // Wait until a long time after bumper bumped?
    d=(d-3)/12;
    if (d<0) d=0;
    utvel*=d;
    urvel*=d;
    acc=10000;
  }
  unsigned char data[MAX_COMMAND_LENGTH];
  // ** workaround for stupid hardware bug, cause unknown, but this works
    // ** with minimal detriment to control
    // ** avoids all values with 1b in highest or 3'rd highest order byte

    // 0x1b is part of the packet terminating string
    // which is most likely what causes the bug

    // ** if 2'nd order byte is 1b, round to nearest 1c, or 1a
    if((urvel&0xff00)==0x1b00){
      // ** if lowest order byte is>127 round up, otherwise round down
      urvel=(urvel&0xffff0000)|((urvel&0xff)>127?0x1c00:0x1aff);
    }

    // ** if highest order byte is 1b, round to 1c, otherwise round to 1a
    if((urvel&0xff000000)==0x1b000000){
      // ** if 3'rd order byte is>127 round to 1c, otherwise round to 1a
      urvel=(urvel&0x00ffffff)|(((urvel&0xff0000)>>16)>127?0x1c000000:0x1aff0000);
    }


    putInt8( 0,                 &(data[0]) );       /* forward motion */
    putInt32( utvel,            &(data[1]) );       /* abs trans velocity*/
    putInt32( acc,     &(data[5]) );       /* trans acc */
    putInt32( STD_TRANS_TORQUE, &(data[9]) );       /* trans torque */
    putInt8( sgn(tvel),         &(data[13]) );      /* trans direction */

    sendCommand(MOT_PORT, 0, MOT_AXIS_SET_DIR, 14, data );
    //printf("Sending com %d%ld\n",sgn(tvel), utvel);
    putInt8( 1,                 &(data[0]) );       /* rotational motion */
    putInt32( urvel,            &(data[1]) );       /* abs rot velocity  */
    /* 0.275 rad/sec * 10000 */
    putInt32( STD_ROT_ACC,      &(data[5]) );       /* rot acc */
    putInt32( STD_ROT_TORQUE,   &(data[9]) );       /* rot torque */
    putInt8( sgn(rvel),         &(data[13]) );      /* rot direction */

    sendCommand(MOT_PORT, 0, MOT_AXIS_SET_DIR, 14, data );
}

void ATRV::sendSystemStatusCommand() {
    sendCommand(SYS_PORT, 0, SYS_STATUS, 0, NULL );
}

void ATRV::setMovement(float tvel_desired, float rvel_desired, float acceleration) {
    double v_current = parserState.v;
    double omega_current = parserState.omega;
    
    double v_error = tvel_desired - v_current;
    double omega_error = rvel_desired - omega_current;
    
    static double v_integral = 0, omega_integral = 0;
    static double v_prev_error = 0, omega_prev_error = 0;
    static ros::Time last_time = ros::Time::now();
    static int omega_reset_counter = 0;
    static const int OMEGA_RESET_INTERVAL = 10; 
    const double MAX_OMEGA_INTEGRAL = 0.3; 


    ros::Time current_time = ros::Time::now();
    double dt = (current_time - last_time).toSec();
    last_time = current_time;
    
    const double Kp = 1.0;   
    const double Ki = 0.1;   
    const double Kd = 0.05;  
    
    double v_p = Kp * v_error;
    v_integral += Ki * v_error * dt;
    double v_d = Kd * (v_error - v_prev_error) / dt;
    
    double omega_p = Kp * omega_error;
    
    omega_reset_counter++;
    if ((omega_reset_counter >= OMEGA_RESET_INTERVAL) or (omega_error * omega_prev_error < 0)) { 
        omega_integral = 0;
        omega_reset_counter = 0;
    }
    omega_integral += Ki * omega_error * dt;
    
    omega_integral = std::max(-MAX_OMEGA_INTEGRAL, 
                             std::min(MAX_OMEGA_INTEGRAL, omega_integral));
    
    double omega_d = Kd * (omega_error - omega_prev_error) / dt;
    double v_command = tvel_desired + v_p + v_integral + v_d;
    double omega_command = rvel_desired + omega_p + omega_integral + omega_d;
    
    v_prev_error = v_error;
    omega_prev_error = omega_error;
    
    setVelocity(v_command * odoDistanceConversion,
                omega_command * odoAngleConversion,
                acceleration * odoDistanceConversion);
}



