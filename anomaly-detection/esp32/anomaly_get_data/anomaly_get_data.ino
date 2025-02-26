#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

const int LED_PIN = 2;
const char* WIFI_SSID = "your_network_here";
const char* WIFI_PASS = "yous_password_here";
const char* SERVER_URL = "http://server_ip_here:4242";

const int SAMPLE_RATE = 200;
const int NUM_SAMPLES = 200;  // 1 second of data at 200Hz
const int I2C_SDA = 21, I2C_SCL = 22;

Adafruit_MPU6050 mpu;
HTTPClient http;

void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize I2C and MPU6050
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) blinkLED(3, 200);
  }
  blinkLED(2, 100);
  
  // Basic sensor config
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
  
  // Connect WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    blinkLED(1, 500);
    Serial.print(".");
  }
  blinkLED(5, 50);
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
}

bool checkServerReady() {
  http.begin(SERVER_URL);
  int httpCode = http.GET();
  bool ready = (httpCode == HTTP_CODE_OK && http.getString() == "1");
  http.end();
  return ready;
}

void sendData(JsonDocument& json) {
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  String jsonString;
  serializeJson(json, jsonString);
  int httpCode = http.POST(jsonString);
  if (httpCode <= 0) Serial.println("Error sending data");
  http.end();
}

void loop() {
  if (!checkServerReady()) {
    delay(100);
    return;
  }
  
  // Prepare JSON
  DynamicJsonDocument json(3 * JSON_ARRAY_SIZE(NUM_SAMPLES) + JSON_OBJECT_SIZE(3));
  JsonArray x_data = json.createNestedArray("x");
  JsonArray y_data = json.createNestedArray("y");
  JsonArray z_data = json.createNestedArray("z");
  
  // Collect data
  unsigned long startTime = millis();
  int samples = 0;
  
  while (samples < NUM_SAMPLES) {
    if (millis() - startTime >= (samples * (1000 / SAMPLE_RATE))) {
      sensors_event_t accel, gyro, temp;
      mpu.getEvent(&accel, &gyro, &temp);
      
      x_data.add(accel.acceleration.x);
      y_data.add(accel.acceleration.y);
      z_data.add(accel.acceleration.z);
      
      if (samples % 50 == 0) {
        Serial.printf("Sample %d: X:%.2f Y:%.2f Z:%.2f\n", 
                     samples, accel.acceleration.x, 
                     accel.acceleration.y, accel.acceleration.z);
      }
      samples++;
    }
  }
  
  // Send data
  digitalWrite(LED_PIN, HIGH);
  sendData(json);
  digitalWrite(LED_PIN, LOW);
  delay(10);
}
