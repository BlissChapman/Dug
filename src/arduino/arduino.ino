
void setup() {
  Serial.begin(9600);
  pinMode(A3, INPUT);
}

void loop() {
  int a3 = analogRead(A3);
  Serial.println(a3);
}
