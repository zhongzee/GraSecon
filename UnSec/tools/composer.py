import random
class SignatureComposer:
    def __init__(self, prompter='a'):
        if prompter not in ['a', 'avg', 'concat', 'isa', 'semantic_hypergraph']:
            raise NameError(f"{prompter} prompter is not supported")

        self._prompter = prompter
        self._composers = {
            'a': self._compose_a,
            'avg': self._compose_avg,
            'concat': self._compose_concat,
            'isa': self._compose_isa,
            'semantic_hypergraph': self._compose_hypergraph,
        }

    def _compose_a(self, signature_list):
        return [f'a {cname}'
                for cname in signature_list]

    def _compose_avg(self, signature_list):
        return [[f'a {catName}' for catName in signature]
                for signature in signature_list]

    def _compose_concat(self, signature_list):
        return ['a ' + signature[0] + ''.join([f' {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def _compose_isa(self, signature_list):
        return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    import random
    def _compose_hypergraph(self, signature_list, num_sentences=3):
        candidate_sentences = []

        def format_list(lst, max_items):
            # 随机选择最多 `max_items` 项，将其转换为字符串，去掉列表符号和引号
            return ', '.join(random.sample(lst, min(len(lst), max_items))) if isinstance(lst, list) else lst

        for signature in signature_list:
            node_name = signature.get('node_name', '')

            # 每个句子随机构造不同的元素组合
            for _ in range(num_sentences):
                # 随机选取每个超边的数量
                attributes_sentence = (
                    f"{node_name} has key biological traits such as {format_list(signature.get('attributes_hyperedge', []), 5)}"
                    if 'attributes_hyperedge' in signature else ""
                )
                candidate_sentences.append(attributes_sentence + ".")

        return candidate_sentences

    # def _compose_hypergraph(self, signature_list, num_sentences=3):
    #     candidate_sentences = []
    #
    #     def format_list(lst, max_items):
    #         # 随机选择最多 `max_items` 项，将其转换为字符串，去掉列表符号和引号
    #         return ', '.join(random.sample(lst, min(len(lst), max_items))) if isinstance(lst, list) else lst
    #
    #     for signature in signature_list:
    #         node_name = signature.get('node_name', '')
    #
    #         # 每个句子随机构造不同的元素组合
    #         for _ in range(num_sentences):
    #             # 随机选取每个超边的数量
    #             attributes_sentence = (
    #                 f"{node_name} has key biological traits such as {format_list(signature.get('attributes_hyperedge', []), 5)}"
    #                 if 'attributes_hyperedge' in signature else ""
    #             )
    #             functional_sentence = (
    #                 f"typically performs roles like {format_list(signature.get('functional_hyperedge', []), 2)}"
    #                 if 'functional_hyperedge' in signature else ""
    #             )
    #             morphological_sentence = (
    #                 f"has a structure characterized by {format_list(signature.get('morphological_hyperedge', []), 2)}"
    #                 if 'morphological_hyperedge' in signature else ""
    #             )
    #             ecological_sentence = (
    #                 f"primarily resides in environments such as {format_list(signature.get('ecological_hyperedge', []), 2)}"
    #                 if 'ecological_hyperedge' in signature else ""
    #             )
    #
    #             # 使用 'and' 连接4个部分
    #             combined_sentence = ' and '.join(
    #                 filter(None,
    #                        [attributes_sentence, functional_sentence, morphological_sentence, ecological_sentence])
    #             )
    #             candidate_sentences.append(combined_sentence + ".")

        return candidate_sentences

    def compose(self, signature_list):
        return self._composers[self._prompter](signature_list)


if __name__ == '__main__':
    composer = SignatureComposer(prompter='semantic_hypergraph')
    signature_list = [
        {
            'node_name': 'Cat',
            'attributes_hyperedge': 'sharp claws, keen eyesight, retractable claws',
            'functional_hyperedge': 'hunting, climbing, stalking prey',
            'morphological_hyperedge': 'whiskers, retractable claws, agile body',
            'ecological_hyperedge': 'domestic environments, wild habitats'
        },
        {
            'node_name': 'Dog',
            'attributes_hyperedge': 'keen sense of smell, loyalty',
            'functional_hyperedge': 'guarding, herding, retrieving',
            'morphological_hyperedge': 'fur coat, non-retractable claws',
            'ecological_hyperedge': 'urban environments, rural settings'
        }
    ]

    result = composer.compose(signature_list)
    print(result)

    # Using 'isa' prompter
    composer_isa = SignatureComposer(prompter='isa')
    signature_list_isa = [
        ['british short hair', 'cat', 'mammal'],
        ['chowchow', 'dog', 'mammal'],
        ['rose', 'flower', 'plant'],
    ]

    result_isa = composer_isa.compose(signature_list_isa)
    print(result_isa)
