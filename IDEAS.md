## Reward Functions
### Brainstorming
*These functions may be used alone or in weighted combination*
2. Dig for blocks attribute values to each block (could be dangerous considering it may opt to dig instead of building houses)

3. Stacking blocks? Preventing from standing too high? How do we get it to place?

4. What would encourage it to get wood?
5. What about food?
6. How do we hierarchy these needs into something that makes sense?

1. Make a discriminator that takes in a screenshot of minecraft and outputs a reward if it's a building and 0 (or negative if it's not). Train an image classifier that takes in the screenshot of minecraft and game pointing at building from many different position. Gives it a binary value on whether or not it's good.

    * How do we collect the dataset?
        1. Take in images from youtube -- is hard because it loses perspective of the ground. We can't assume the agent is gonna see the same perspective when out in the field.
        2. Take in images from an automated image collector. Feed Maps, centered on Houses. Explore the outside of house. Return images of the house.
            * This could also be a good policy to evaluate the actual goodness of a structure made.
            * How would you define whether a structure has been made? Should you learn that as well?
## Block Rewards
I made a CSV called block rewards that lays out a possible reward value for each block. Ideally we combine this with things that give a higher reward so that it is encouraged to do something else other than just dig. 

Maybe this reward could be learned somehow?
