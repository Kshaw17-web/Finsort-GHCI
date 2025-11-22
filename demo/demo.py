from finsort.inference import predict_category
from finsort.explain import explain_prediction
import os

def interactive_demo():
    print("FinSort interactive demo. Make sure you ran training to create model.pkl and vectorizer.pkl.")
    while True:
        text = input('\nEnter transaction (or type exit): ')
        if text.strip().lower() in ('exit', 'quit'):
            break
        result = predict_category(text)
        print('\nPrediction:')
        for k,v in result.items():
            print(f"  {k}: {v}")
        print('\nTop indicators: ', explain_prediction(text))
        if result['confidence'] < 0.6:
            correct = input('Low confidence. Correct category if wrong (press enter to skip): ').strip()
            if correct:
                with open(os.path.join(os.path.dirname(__file__), '..', 'finsort', 'feedback.log'), 'a') as f:
                    f.write(f"{text},{correct}\n")
                print('Logged feedback.')
if __name__ == '__main__':
    interactive_demo()
