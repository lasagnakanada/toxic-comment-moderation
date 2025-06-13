# toxic-comment-moderation

After EDA it was observed that amount of comments which could are toxic in general is 22%. By splitting all toxic comments into separate classes as:

'toxic' 
'severe_toxic'
'obscene' 
'threat'
'insult' 
'identity_hate'. 

We get more concrete information about how specifically each comment is toxic (if it is).

Most of the comments have significant disbalance in terms of its length, belonging to a specific class (there is only 5% of threat comments for example).
This information is useful for future analysis and input limitations in our model.

During investigation of our data I decided that F1-score is highly important metric for this kind of task where it is highly important to predict toxic comments (obviously) and still be able to precisely detect non-toxic ones. What I mean by this is that you can just mark all comments as toxic and get recall 100% but we still want to accept non-toxic comments in our moderation to provide an option for users to share their opinion with others. With such understanding of existing balance demand, F1-score is a highly reliable metric which can describe both: ability to detect 'niche' data and still getting high perfomance on usual data-traffic. 

P.S.: The same logic applies to all tasks where it is highly important to detect low-frequently appearing event and still soberly deal with rest of the data. Paradox is that soon your main task becomes to properly detect 'rest of the data' and not those rarely appearing events since you can just always predict any event as 'rare' and get 100% recall. Think about medical diagnosis process, food for thought.


input_size = 10000 (at this point we use TF-IDF to select 10k most frequently appearing words and describe them as a single vector where each dimension represents single weight of a particular word)

hidden_size = 256 (in future this number might increase after replacing TF-IDF with something more complex but currently it is just important to move in direction of our aim)

output_size = 6 (we have six classes to classify from, for each output unit we get a binary classifier which decides whether specific comment is toxic or not).






