from tqdm import tqdm
import random

def solution2(U, weight):
    # no two consecutive elements sum to > U
    # no element alone > U
    # use a stack to implement

    # use array to implement python stack
    bridge = []

    # first car on stack
    bridge.append(weight[0])
    numcars = len(weight)

    for i in range(1, numcars):
        newcar = weight[i]
        onbridge = bridge[-1]

        # if under weight, add new car to bridge
        # otherwise keep lighter of two
        if (onbridge + newcar) <= U:
            bridge.append(newcar)
        else:
            newcar = min(newcar, onbridge)
            bridge[-1] = newcar

    turnaround = numcars - len(bridge)

    # case: all cars need to turn
    if bridge[-1] > U:
        turnaround = numcars

    return turnaround

def solution(U, weight):
    # Implement your solution here

    if len(weight) < 2 and weight[-1] <= U:
        return 0

    right_pointer = 1
    left_pointer = 0
    turn_arounds = 0

    while right_pointer < len(weight):
        if weight[left_pointer] + weight[right_pointer] > U:
            turn_arounds += 1

            if weight[left_pointer] > weight[right_pointer]:
                left_pointer = right_pointer
        else:
            left_pointer = right_pointer

        right_pointer += 1

    if weight[left_pointer] > U:
        turn_arounds += 1

    return turn_arounds

# Run the test cases with progress tracking using tqdm
for _ in tqdm(range(1000000), desc="Running Tests"):
    U = random.randint(1, 100)  # Random U between 1 and 100
    weight = [random.randint(1, 100) for _ in range(random.randint(1, 5000))]  # Random weights between 1 and 5000

    assert solution(U, weight) == solution2(U, weight), f"Test failed for U={U} and weight={weight}"