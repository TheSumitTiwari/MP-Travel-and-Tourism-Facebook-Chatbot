# MP Travel and Tourism Facebook Chatbot

The main objective for this project is to develop a smart chatbot that will be capable of engaging in conversation with general people similar to a human being and will help the tourists to guide them and provide information about different tourist places in MP to help them have a smooth journey throughout. We have hosted this chatbot on Facebook Messenger app using Flask Server as a middleware.

## Getting Started

Firstly you need to Setup Facebook Developer Account on [Meta for Developers](https://developers.facebook.com/), Then create a Facebook App & Facebook page, and set it all up.
Fetch your PAGE_ACCESS_TOKEN and also add a VERIFY_TOKEN. [Click on this link for detailed tutorial.](https://developers.facebook.com/docs/messenger-platform/getting-started/quick-start/)

### Installation

Download or Clone the repo, Navigate to the directory containing the files and run
` python app.py install` to install the dependencies.
` python app.py` to run the project.

### Use

Copy your PAGE_ACCESS_TOKEN and VERIFY_TOKEN and insert in `app.py` where specified.
Use [ngrok](https://ngrok.com/) to expose local webserver to the internet so that it can be used for callback verification needs to be done for using a webhook with Facebook app.

**Individual Modules of this project can be found here:**

1. [Facebook messenger bot using flask server](https://github.com/TheSumitTiwari/Facebook-messenger-bot-using-flask-server)
2. [Madhaya Pradesh Travell and Tourism Chatbot Dataset](https://github.com/ShilpiKiran/MP-travel-and-tourism-chatbot-dataset)
3. [Spelling Corrector](https://github.com/ShilpiKiran/Spelling-Corrector)
