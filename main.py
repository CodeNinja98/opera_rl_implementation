import logging
import coloredlogs

import gym


coloredlogs.install(logging.DEBUG)

def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = gym.make("LunarLander-v2", render_mode="human")


    for i in range(10):
        obs = env.reset()
        done = False
        for step_counter in range(100):

            # Step your model here.
            # Currently, it's doing random actions
            random_act = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(random_act)

            if done:
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()