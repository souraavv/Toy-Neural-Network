set title "Stochastic Gradient Descent"

set ylabel "Loss Function"
set xlabel "Number of Epochs"

set datafile separator ","
plot 'c_stoc.csv'  using 1:2 with linespoint title "SGD"

pause -1 "press key"                       