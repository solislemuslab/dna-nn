## Neural network model for pseudomonas image data
## needs to be read inside fit-nn.jl
## Claudia July 2019


## -------------------------------------------
## function to keep track of all models tested
## -------------------------------------------
function whichModel(num::Integer)
    if num == 1
        ## based on mlp.jl
        model = Chain(
                      Dense(p,32,relu),
                      Dense(32,2),
                      softmax
                      )

        loss(x, y) = crossentropy(model(x), y)

        optimiser = ADAM(0.00001)
        return model,loss,optimiser
    end
end





## model = Chain(
##     Conv((2, 2), p=>div(p,100), relu),
##     x -> maxpool(x, (2,2)),
##     Conv((2, 2), div(p,100)=>div(p,1000), relu),
##     x -> maxpool(x, (2,2)),
##     Dense(div(p,1000), 2),
##     softmax
## )

##loss(x, y) = crossentropy(model(x), y) ## fixit: change to account for unbalancedness
##loss(x, y) = crossentropy(softmax(model(x)), y)




