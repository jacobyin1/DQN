# DQN
 Deep q network for blockade game. 

I followed the double deep q network described here 

https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

and I implemented the Q(s, a, o) two agent zero sum game reinforcement learning algorithm  here.

https://courses.cs.duke.edu/spring07/cps296.3/littman94markov.pdf

# Model description

The model takes in flattened array of zeros and ones according to whether an obstacle is in the space, and the character positions are appended in binary. The output is a 4x4 array where entry i, j is the estimated value of action i and opponent action j. The model updates according to max over i, min over j of the q values of the next state passed through the target equation, with the error from bellman equation.


# To do:

1. Work on fine tuning parameters and see how good this kind of architecture can work.
2. Experiment with larger models such as more layers.
3. Compare with normal q learning.
4. Try another method.

   



https://github.com/user-attachments/assets/a78dd19f-9211-4cf9-9ebe-7a3a11dcf610


Here two snakes play against each other according to the same trained model.

![Figure_1](https://github.com/user-attachments/assets/64e17bf4-22f3-474c-838a-9c4d5be48893)

Loss graph (tick upwards was from adding the greater than one term to loss function)
