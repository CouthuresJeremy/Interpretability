mkdir -p results
scp -r jecouthu@lxplus.cern.ch:/eos/user/j/jecouthu/Interpretability/conditional_entropy/mutual_information_event*_layer*_range*.csv ./results

python merge_layer_mutual_information_neuron_range.py