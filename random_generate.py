import random

def generate_random_number(length=10):
    # 确保生成的数字以非零开头
    first_digit = random.randint(1, 9)
    # 剩余的数字可以是 0 到 9
    other_digits = [str(random.randint(0, 9)) for _ in range(length - 1)]
    return str(first_digit) + ''.join(other_digits)

# 生成一个随机的10位数字
random_number = generate_random_number()
print(random_number)
