void setup() {
//  analogReference(EXTERNAL);
  Serial.begin(9600);
  pinMode(A2, INPUT);
}

void loop() {
  int a0 = analogRead(A2);
  Serial.println(a0);
}
