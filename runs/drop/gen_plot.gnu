set terminal postscript
set output '| ps2pdf - output.pdf'
set boxwidth 0.5
set style fill solid
set xlabel 'Dropped feature map'
set ylabel 'Avg. score'
set key off
plot "breakout1_layer0" with boxes

