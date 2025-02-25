#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

const int LED_PIN = 2;
const char* WIFI_SSID = "your_network_here";
const char* WIFI_PASS = "your_password_here";
const char* SERVER_URL = "http://your_api_ip_here:8000/predict";  // Endpoint da API de inferência

const int SAMPLE_RATE = 200;
const int NUM_SAMPLES = 100;  // 0.5 segundos de dados em 200Hz
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
    
    // Inicializa I2C e MPU6050
    Wire.begin(I2C_SDA, I2C_SCL);
    if (!mpu.begin()) {
        Serial.println("MPU6050 não encontrado!");
        while (1) blinkLED(3, 200);
    }
    blinkLED(2, 100);
    
    // Configuração básica do sensor
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
    
    // Conexão WiFi
    Serial.print("Conectando ao WiFi");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        blinkLED(1, 500);
        Serial.print(".");
    }
    blinkLED(5, 50);
    Serial.printf("\nConectado! IP: %s\n", WiFi.localIP().toString().c_str());
}

void sendData(JsonDocument& json) {
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  String jsonString;
  serializeJson(json, jsonString);
  
  int httpCode = http.POST(jsonString);
  
  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String response = http.getString();
      
      // Parse da resposta da API
      DynamicJsonDocument responseDoc(1024);
      deserializeJson(responseDoc, response);
      
      bool isAnomaly = responseDoc["is_anomaly"].as<bool>();
      float distance = responseDoc["distance"].as<float>();
      
      // Feedback visual
      if (isAnomaly) {
        blinkLED(3, 100);  // Pisca 3x rápido se for anomalia
        Serial.println("Anomalia detectada!");
      }
    }
  } else {
    Serial.println("Error sending data");
  }
  http.end();
}

void loop() {
  // Prepare JSON (mudança no formato para matriz 2D)
  DynamicJsonDocument json(16384);
  JsonArray data = json.createNestedArray("data");
  
  // Collect data
  unsigned long startTime = millis();
  int samples = 0;
  
  while (samples < NUM_SAMPLES) {
    if (millis() - startTime >= (samples * (1000 / SAMPLE_RATE))) {
      sensors_event_t accel, gyro, temp;
      mpu.getEvent(&accel, &gyro, &temp);
      
      // Criar array [x,y,z] para cada amostra
      JsonArray sample = data.createNestedArray();
      sample.add(accel.acceleration.x);
      sample.add(accel.acceleration.y);
      sample.add(accel.acceleration.z);
      
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
  delay(100);
}