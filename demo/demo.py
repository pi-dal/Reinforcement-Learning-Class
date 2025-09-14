import gym
import reinforcement_learning_class.env

def main():
    env = gym.make("GridWorld-v0", size=5, render_mode="human")

    obs, info = env.reset(seed=42)
    total_reward = 0

    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("Episode finished after {} steps".format(t + 1))
            break

    env.close()
    print("Total reward:", total_reward)

if __name__ == "__main__":
    main()
