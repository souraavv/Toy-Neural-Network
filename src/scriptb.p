set title "Batch Gradient Descent"

set ylabel "Loss Function"
set xlabel "Number of Epochs"

set datafile separator ","
plot 'c_batch.csv'  using 1:2 with linespoint title "BGD"

pause -1 "press key"                       