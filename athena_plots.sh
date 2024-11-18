mkdir -p activations
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/activations/* ./activations
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/activations/activations_event000000101.pt ./activations
mkdir -p model
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/artifacts/*best* ./model
#mkdir -p AcornPlots
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/*.png ./AcornPlots

mkdir -p csv
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/permutation_importance/* ./csv
scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/metric_learning/input_data/*101* ./csv
#scp -r ../Interpretability jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/
#scp ./pykan/kan_hep_metric_learning.py jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/Interpretability/pykan
mkdir -p data
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/athena_100_events/*101* ./data
#scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/acorn/acorn/examples/Example_2/data/Example_2/feature_store/**/*101* ./data
