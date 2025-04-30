# Implementation of PER and Dueling DQN 
----------------

### Implementations of Prioritized Experience Replay


For the Proportinal PER I use a sum-tree structure as mentioned in the paper. 

For the Rank-Based PER I used a different (somewhat-original) implementation based on `SortedList` from `sortedcontainers`. The data structure is essentially a dictionary but is ranked based on the value (absolute TD-error). This is done by sorting the errors using the `SortedList` and bookeeping the respective indices (for the samples stored in the buffer). The runtime (based on the `sortedcontainers` docs) is O(log(n)) for updating and retrieving and we don't run into the problem that of unbalanced arrays, as in the original paper.   

-------