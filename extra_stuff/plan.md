What the plan is

Optimizers
- General Introduction: https://arxiv.org/pdf/1609.04747
- SGD: Nestorov, Momentum, L2, and Damp: 
    - https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
- RMSProp
    - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- AdaGrad:
    - https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
- Adam
    - https://arxiv.org/pdf/1412.6980
- AdamW
    - https://arxiv.org/pdf/1711.05101
- LAMB
    - https://arxiv.org/pdf/1904.00962


Normalization
- Batch Normalization: 
    - https://arxiv.org/abs/1502.03167
- Layer Normalization:
    - https://arxiv.org/abs/1607.06450
- Dropout
    - https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf




TODO
- Need to get full test coverage
- Need to implement where functionality, just due to it being usefull in Huber
- I want the framework to feel chainy, so you don't use Objects for anything other than layer stuf. also a Sequential. All of the class stuff is essentially going to be extensions for the class.



Need to implement Reduction on all the losses

Need to implement Huber Loss: this is the final loss to implement!!!