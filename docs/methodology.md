# Methods

## Near-Duplicate Detection

We performed near duplicate detection using minhash lsh.

## Gender Classification

We classified the gender in different ways. 

First, we used `nomquamgender`, a python module for gender prediction. Second, for Chinese Pinyin names, we used the module `chgender`. Both modules work when the name is in the format "firstname lastname" or "firstname_lastname" or similar, with a seperator.

For the names that were not captured by this, i.e. usernames of the format "firstnamelastname" or usernames that do not contain a firstname, we manually looked at the usernames. More specifically, for all names that were in "firstnamelastname" (N=XXX), we added their username to the list of usernames. 

For the remaining users, we checked everyone who wrote five or more posts for identifiers of their gender. This took different ways: some users refer ot their personal website, blog, or twitter account where we could see their name or a gender identification; others referred to themselves in their description as "male"/"guy" or "female"/"girl" or said that they don't identify with genders, etc. Lastly, we checked in the comments in which this user was mentioned for pronouns and, if the user didn't object to that, took these pronouns. The rationale here is that either the commenting user and the user under question have met in real life and so know of each other's gender or the assumed it and the user under question accepted this assumption.

This results is a variety of as "unknown" gender identified users: Deleted accounts, everyone who does not identify as male or female, everyone who does not identify at all, ambiguous name holders, and group accounts.

We want to note that gender classification is still noisy from different aspects. People may change their gender and still have their old username. Different cultures may use a female or multi-gender name as male one and vice versa. People may have been referred to by others using the wrong gender assumptions and didn't raise the issue for several reasons. However, we think these are reasonable levels of noise and accept them in sake for this classification.

## Links

We extracted all links in the posts unless it is a linkpost (a post that only contains one link to another website). For forum posts, we idenftified arXiv papers to infer the arXiv DOI over the openalex API for a cleaner graph analysis.