To run my project:

	$ python3 perceptron.py --train_file [data] --iterations [k] --train_rate [rate]

The function has been modify such that training rate can be specify optionally, but the default value is 1. After running on data set, the program will output the below format:

	$ python3 perceptron.py --train_file data/challenge1.dat --iterations 5000

	---------------------- training ---------------------
	size of data:  4000
	iterations 5000

	Converged at: 4155th iterations
	---------------------- result ---------------------
	Train accuracy: 1.0
	Feature weights (bias last): -7.041481377055892 -453.30486991359084 86.16504987953 -332.0175379613585 -490.44357166395827 84.87768725208167 -144.39624354313548 341.2220542188988 -579.8147681657225 370.2562733232363 82.04546960620826 496.4669646235532 57.42237160757858 138.21386925616684 -446.33988590094174 -89.17189695750639 -287.0977355484778 545.6231647743659 41.39053301201871 4.73896807210497 -433.97988078754344 -105.51492609600274 377.2512436060767 -35.94262989968571 318.217904550593 -260.7824683872749 380.94479414161617 -35.9815221205802 -313.675185822852 -32.11367156285925 120.0
	R (maximun vector norm):  2.896098539414707
	δ^2 (minimun seperation):  0.00032012259506003114
	(R^2)/k = 6.135885079081744e-05

If the data is non-separable, (R^2)/k will be inf since the data does not converge at any k value.
