from pyarc import TransactionDB, CBA
import pandas as pd
from pyarc.data_structures.car import ClassAssocationRule
from data.preprocess import prompt_set
import semantics

# pyARC: https://github.com/jirifilip/pyARC

mined_rules = []

def rule_to_em_rule(rule: ClassAssocationRule):
    em_rule = ''
    # print(rule.antecedent.itemset)
    for key, val in rule.antecedent.itemset.items():
        em_rule += f'{key}({val}) & '
    cons = rule.consequent
    em_rule = f'{em_rule[:-3]} → {cons.attribute}({cons.value}) [conf={round(rule.confidence, 2)},supp={round(rule.support, 2)}]'
    return em_rule


def train_model(data, support, confidence, rule_length):
	print('Mining rules...')
	headers = data.columns
	data = TransactionDB.from_DataFrame(data)

	cba = CBA(support=support, confidence=confidence, maxlen=rule_length, algorithm="m1")
	cba.fit(data)
	accuracy = cba.rule_model_accuracy(data)

	print('Determining semantic coherence...')
	semantics.precompute_attributes(headers)

	for rule in cba.clf.rules:
		antecedent = list(rule.antecedent.itemset.keys())
		coherence = semantics.get_rule_coherence(antecedent)
		mined_rules.append((f'{rule_to_em_rule(rule)}',rule.confidence, coherence,
			rule.consequent.value))
		
	print_model()
	print(f'Model accuracy: {round(accuracy * 100, 2)} %')
	print(f'Model support: {cba.support}')

# pred = cba.predict(TransactionDB.from_DataFrame(data_train.loc[[0]]))
# print(pred)
def prompt_rule_length():
	while True:
		num = input('Select the maximum rule length: ')
		try:
			num = int(num)
			if num < 2:
				print('The rule length must be at least 2!')
				continue
			return num
		except ValueError:
			print('That is not a correct number!')


def prompt_float(msg: str):
	while True:
		num = input(msg)
		try:
			num = float(num)
			if num < 0 or num > 1:
				print('The number must be in the [0.0, 1.0] interval!')
				continue
			return num
		except ValueError:
			print('That is not a correct number!')


def print_model():
	print('')
	for rule in mined_rules:
		print(f'{rule[0]} SC: {round(rule[2], 2)}')
	print('')


def calc_heuristic(alpha, SH, CH):
	return alpha * SH + (1 - alpha) * CH


def reorder_results(alpha):
	global mined_rules
	mined_rules.sort(key=lambda x: calc_heuristic(alpha, x[2], x[1]), reverse=True)


def prompt_reorder():
	while True:
		answer = input('Reorder the result? (Y/N): ').strip().lower()

		if answer not in ('y', 'n'):
			print("Valid responses are only 'y' or 'n'!")
			continue

		if answer == 'n':
			break

		alpha = prompt_float('Select alpha value [0.0, 1.0]: ')
		reorder_results(alpha)
		print_model()


def prompt_filter():
	global mined_rules

	while True:
		answer = input('Filter the result based on consequent? (Y/N): ').strip().lower()

		if answer not in ('y', 'n'):
			print("Valid responses are only 'y' or 'n'!")
		else:
			break

	if answer == 'n':
		return

	target = input('Enter the desired consequent value: ').strip().lower()
	mined_rules = list(filter(lambda x: x[3].strip().lower() == target, mined_rules))

	if mined_rules:
		print_model()
	else:
		print('No such rules found!')


def main():
	global mined_rules
	while True:
			mined_rules = []
			train_model(prompt_set(),
				prompt_float('Please select desired model support [0.0, 1.0]: '),
				prompt_float('Please select desired model confidence [0.0, 1.0]: '),
				prompt_rule_length())
			prompt_filter()
			if mined_rules:
				prompt_reorder()


if __name__ == '__main__':
	semantics.load_model('en')
	try:
		main()
	except KeyboardInterrupt:
		print('\nBye!')
