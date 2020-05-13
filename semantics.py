import gensim
from typing import List
from itertools import product, combinations

# Global variable storing the word embedding
model = None

# All the possible tokens (for the English model only)
tokens = ['_NOUN', '_VERB', '_ADJ', '_ADV', '_NUM', '_PROPN', '_X', '_SYM', '_INTJ']

# Used to store precomputed attribute coherence
attribute_coherence = {}

def _find_instances_of_word(word: str) -> List[str]:
	instances = []
	for token in tokens:
		full_word = word + token
		if full_word in model.vocab:
			instances.append(full_word)
	return instances


def _get_similarity(word1: str, word2: str):
	word1, word2 = word1.lower(), word2.lower()
	instances = (_find_instances_of_word(word1), _find_instances_of_word(word2))
	if not (instances[0] and instances[1]):
		raise Exception(word1 if not instances[0] else word2 + ' is not in the gensim model vocabulary!')
	
	max_similarity = 0
	combinations = product(instances[0], instances[1])

	for combination in combinations:
		similarity = model.similarity(combination[0], combination[1])
		max_similarity = max(max_similarity, similarity)

	return max_similarity


def _get_attribute_coherence(att1: str, att2: str) -> float:
	att1, att2 = att1.split(), att2.split()
	similarity = 0

	combinations = list(product(att1, att2))
	for combination in combinations:
		similarity += _get_similarity(combination[0], combination[1])

	return similarity / len(combinations)


def precompute_attributes(header: List[str]) -> None:
	att_pairs = combinations(header, 2)

	for pair in att_pairs:
		attribute_coherence[pair] = _get_attribute_coherence(*pair)


def get_rule_coherence(rule: List[str]) -> float:
	"""Determines the semantic coherence across the rule.
	If the rule consists of only one attribute, the result is as follows:

	- if the attribute contains only one word, the resulting coherence is 1.0
	- otherwise the coherence is determined as the average similarity between individual
		word-pairs in the attribute
	"""
	if not rule:
		raise Exception('Cannot determine semantic coherence of an empty rule.')

	precomputed = True

	if len(rule) == 1:
		rule = rule[0].split()
		if len(rule) == 1:
			return 1.0
		else:
			precomputed = False

	att_pairs = list(combinations(rule, 2))
	coherence = 0

	for pair in att_pairs:
		if precomputed and pair not in attribute_coherence:
			swapped = pair[::-1]
			if swapped not in attribute_coherence:
				raise Exception(f'Pair {pair} not found in the precomputed coherence dictionary.'
					+ ' Forgot to run attribute precomputing first?')
			coherence += attribute_coherence[swapped]
		else:
			coherence += attribute_coherence[pair] if precomputed else _get_attribute_coherence(*pair)

	return coherence / len(att_pairs)


def _test(language):
	test_sets = {
		'en': ['cat', 'dog', 'car', 'cat shelter', 'poodle'],
		'cz': ['kočka', 'pes', 'lopata', 'kočičí útulek', 'pudl']
	}
	word_list = test_sets[language]

	# Testing purposes
	precompute_attributes(word_list[:3])
	c1 = get_rule_coherence(word_list[:2]) # should be relatively high
	c2 = get_rule_coherence(word_list[:3]) # should be lower compared to previous
	c3 = get_rule_coherence([word_list[0]]) # should be 1.0
	c4 = get_rule_coherence([word_list[3]]) # should be something else than 1.0

	tests = [c1 > c2, c3 == 1.0, c4 < 1.0]

	for i in range(len(tests)):
		if not tests[i]:
			print(f'TEST {i + 1} FAILED')

	try:
		get_rule_coherence([word_list[0], word_list[4]]) # should raise an exception
		print('EXCEPTION TEST FAILED')
	except:
		print('ALL TESTS PASSED')


def load_model(lang: str) -> None:
	"""Loads the model corresponding to the provided language code into memory."""
	global model, tokens

	if lang == 'cz':
		tokens = ['']

	# Another great gensim-compatible possibility are fastText embeddings
	# provided by Facebook at https://fasttext.cc/
	print('Loading word embeddings...')
	filepath = f'{__file__}/../word_embeddings/{lang}/model.bin'
	try:
		model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)
	except:
		print(f'Could not find the precomputed word embedding at {model_path}.'
			+ ' The model can be downloaded from http://vectors.nlpl.eu/explore/embeddings/en/models/')
		exit()
	print('Loaded.')


def _prompt_lang():
	while True:
		lang = input('Please specify vocabulary language (cz/en): ')
		if lang not in ('cz', 'en'):
			print('That is not an acceptable language code!')
		else:
			break

	load_model(lang)
	return lang


if __name__ == '__main__':
	_test(_prompt_lang())

	# Can be used manually to compute semantic coherence between certain set of words
	while True:
		try:
			rule = input('Please enter comma-separated rule attribute names: ')
			rule = [att.strip().replace('"', '') for att in rule.split(',')]
			precompute_attributes(rule)
			print(get_rule_coherence(rule))
		except KeyboardInterrupt:
			print('\nBye!')
			exit()
		except Exception as err:
			print(err)
