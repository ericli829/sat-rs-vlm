# Counting Expert capacity parameter counts

| capacity | per tap | four merger taps |
|---|---:|---:|
| W512 D1 | 6,040,064 | 24,160,256 |
| W512 D2 | 6,307,840 | 25,231,360 |
| W768 D1 | 9,254,400 | 37,017,600 |
| W768 D2 | 9,852,672 | 39,410,688 |
| W1024 D1 | 12,599,808 | 50,399,232 |
| W1024 D2 | 13,659,648 | 54,638,592 |

The categorical count head adds 1,324,560 parameters. C3 interface LoRA adds 1,310,720.
Width is the first capacity axis; D2 is reserved for the second-stage depth study.
