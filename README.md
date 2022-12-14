# 591-RTDL

Code and Result proofs can be found in the 'proj' directory.

Result proofs can be viewed as screenshots in folders that end with 'SC'.

Complete Results for 2 Convolutional Layer Network:

| Approach | Var(0)    | Var(0.05)    | Var(0.075)    | Var(0.1)    | Var(0.125)    | Var(0.15)    | Var(0.2)    | Var(0.3)    | Var(0.4)    | Var(0.5)    | Var(0.6)    | Var(0.7)    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unpruned | 63   | 57   | 55   | 52   | 50   | 48   | 44   | 38   | 34   | 31   | 29   | 27   |
| Vanilla Pruned | 62   | 52   | 48   | 45   | 42   | 40   | 36   | 31   | 27   | 25   | 23   | 22   |
| Pruned New Objective | 61   | 56   | 52   | 50   | 48   | 45   | 41   | 36   | 33   | 30   | 28   | 26   |
| Pruned Train Noise | 62   | 53   | 49   | 45   | 42   | 39   | 35   | 30   | 27   | 24   | 23   | 21   |
| Pruned Input Noise | 57   | 60   | 60   | 60   | 59   | 57   | 55   | 50   | 45   | 41   | 37   | 34   |

Complete Results for 4 Convolutional Layer Network:

| Approach | Var(0)    | Var(0.05)    | Var(0.075)    | Var(0.1)    | Var(0.125)    | Var(0.15)    | Var(0.2)    | Var(0.3)    | Var(0.4)    | Var(0.5)    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:|
|Unpruned |76| 50| 42| 36| 33| 30| 27| 22| 20| 18|
| Vanilla Pruned | 75| 45| 38| 33| 29| 27| 23| 19| 18| 17|
| Pruned New Objective | 76| 48| 41| 36| 33| 30| 27| 22| 20| 18|
| Pruned Train Noise |75| 44| 37| 32| 29| 27| 24| 20| 19| 17|
| Pruned Input Noise |65| 69| 69| 69| 68| 67| 64| 58| 53| 48|

Complete Results for ResNet18:

| Approach | Var(0)    | Var(0.05)    | Var(0.075)    | Var(0.1)    | Var(0.125)    | Var(0.15)    | Var(0.2)    | Var(0.3)    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unpruned |83| 43| 36| 31| 28| 25| 23| 19|
| Vanilla Pruned | 84| 33| 24| 19| 16| 15| 14| 14|
| Pruned New Objective | 85| 44| 35| 30| 26| 24| 21| 19|
