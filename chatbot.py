def chatbot_response(user_input):
    # Convert user input to lowercase for easier matching
    user_input = user_input.lower()

    # Define responses based on user input
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"
    elif "how are you" in user_input:
        return "I'm just a computer program, but thanks for asking! How can I help you?"
    elif "what is your name" in user_input:
        return "I am a simple chatbot created to assist you."
    elif "how is your day today" in user_input:
        return "a nice day for me to interact with you"
    elif "help" in user_input:
        return "Sure! What do you need help with?"
    elif "what is the reason to interact with me" in user_input:
        return "generate an documentation for a given topic"
    elif "thank you" in user_input:
        return "you are welcome for the next time"
    elif "bye" in user_input or "exit" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

def main():
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
