const { google } = require('googleapis');

// Initialize the YouTube API client
const youtube = google.youtube({
    version: 'v3',
    auth: 'AIzaSyAecqemsoC0irtkDMJJSDZf2MItDkV5jDc' // Replace 'YOUR_API_KEY' with your actual API key
});

// Function to search YouTube videos by keyword
async function searchYouTubeVideos(keyword) {
    try {
        // Make an API call to the YouTube 'search.list' method
        const response = await youtube.search.list({
            part: 'snippet',
            q: keyword,
            maxResults: 10, // You can change the number of results per page
            type: 'video'   // You can include 'channel' or 'playlist' if needed
        });

        // Log the response with video details
        console.log(response.data.items);
    } catch (error) {
        console.error('Error during YouTube API call:', error);
    }
}

searchYouTubeVideos('Ababil-2 UAV sound');