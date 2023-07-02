set title "Mini-Batch Gradient Descent"

set ylabel "Loss Function"
set xlabel "Number of Epochs"

set datafile separator ","
plot 'c_minibatch.csv'  using 1:2 with linespoint title "MBGD"

pause -1 "press key"                       