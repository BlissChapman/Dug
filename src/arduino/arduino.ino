
void setup() {
  Serial.begin(9600);
  pinMode(A2, INPUT);
}

void loop() {
  int a2 = analogRead(A2);
  Serial.println(a2);
}
