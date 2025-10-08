import random

red_balls = 3
blue_balls = 4
black_balls = 2

def dice_roll():
    global red_balls, blue_balls, black_balls
    roll_result = random.randint(1, 6)
    print("Dice roll result: ", roll_result)
    if roll_result in (2, 3, 5):
        print("We add a black ball")
        black_balls += 1
    elif roll_result == 6:
        print("We add a red ball")
        red_balls += 1
    else:
        print("We add a blue ball")
        blue_balls += 1
    print('\n')

def prob_red_ball():
    total_nr_balls= red_balls + black_balls + blue_balls
    return red_balls/total_nr_balls

def run():
    dice_roll()
    print("Red Balls: ", red_balls, '\n')
    print("Blue Balls: ", blue_balls, '\n')
    print("Black Balls: ", black_balls, '\n')

    print("Probability for red ball is: ", prob_red_ball())

run()

#Bonus
def prob_red_ball_theoretical():
    initial_red = 3
    initial_blue = 4
    initial_black = 2
    total_prob = 0
    for roll in range(1, 7):
        red = initial_red
        blue = initial_blue
        black = initial_black
        if roll in (2, 3, 5):
            black += 1
        elif roll == 6:
            red += 1
        else:
            blue += 1
        total = red + blue + black
        total_prob += red / total
    return round(total_prob / 6, 3)

print("Probabilitate teoretică:", prob_red_ball_theoretical())


    






