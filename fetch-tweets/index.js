require('dotenv').config();

const { TwitterApi } = require('twitter-api-v2');

const twitterClient = new TwitterApi({
    appKey: process.env.TWITTER_API_KEY,
    appSecret: process.env.TWITTER_API_SECRET_KEY,
    accessToken: process.env.TWITTER_ACCESS_TOKEN,
    accessSecret: process.env.TWITTER_ACCESS_TOKEN_SECRET,
});


const readOnlyClient = twitterClient.readOnly;

// Function to search tweets by a keyword
async function searchTweets(keyword) {
    try {
        // Search for recent tweets containing the keyword 'Ababil'
        const response = await readOnlyClient.v2.search(keyword, { max_results: 10 });
        // Display the results
        console.log(response.data);
    } catch (error) {
        console.error(error);
    }
}


searchTweets('Ababil');