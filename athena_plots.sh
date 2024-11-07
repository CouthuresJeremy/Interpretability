mkdir -p activations
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/activations/* ./activations
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/activations/activations_event000000101.pt ./activations
mkdir -p model
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/artifacts/*best* ./model
#mkdir -p AcornPlots
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/*.png ./AcornPlots

mkdir -p csv
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/permutation_importance/* ./csv
scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/input_data/input_data_event000000101.csv ./csv

