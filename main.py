import gymnasium as gym
from Agent import Agent
EP_LEN = 100
NUM_EPS = 5

def main():
    env = gym.make("LunarLander-v2", render_mode="human",enable_wind=True)
    agent = Agent(EP_LEN)
    for i in range(NUM_EPS):
        obs = env.reset()[0]
        reward = 0
        done = False
        agent.beginEpisode()
        for step_counter in range(EP_LEN):
            act = agent.step(obs,reward)
            obs, reward, done, truncated, info = env.step(act)
            if done:
                break
        agent.endEpisode()
        print(f"[{i}] Episode complete")
    env.close()

if __name__ == "__main__":
    main()