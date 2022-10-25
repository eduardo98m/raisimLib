
def obs_dict_parser(obs_dict, state_variables_dict):
    """Parse the obs_dict from the raisimGym environment.

    Args:
        obs_dict (dict): The observation dictionary from the raisimGym environment.
        state_variables_dict (dict): The state variables dictionary from the cfg.yaml.

    Returns:
        dict: A dictionary with the parsed observation dictionary.
    """
    # Parse the observation dictionary

    encoder_first_index = -1
    encoder_last_index = 0

    non_privileged_first_index = -1
    non_privileged_last_index = 0

    privileged_first_index = -1
    privileged_last_index = 0
    
    for observation in state_variables_dict:
        obs_info = state_variables_dict.get(observation)

        if obs_info and obs_info["enabled"]:
            if obs_info.get('encoder_obs'):
                if encoder_first_index == -1:
                    encoder_first_index = obs_dict[observation][0]
                encoder_last_index = obs_dict[observation][-1]

            if obs_info.get('priviledge'):
                if privileged_first_index == -1:
                    privileged_first_index = obs_dict[observation][0]
                privileged_last_index = obs_dict[observation][-1]
       
            else:
                if non_privileged_first_index == -1:
                    non_privileged_first_index = obs_dict[observation][0]
                non_privileged_last_index = obs_dict[observation][-1]

        else:
            print("The observation {} is disabled in the state_variables_dict.".format(observation))


    obs_dict['encoder']        = [encoder_first_index, encoder_last_index]
    obs_dict['non_privileged'] = [non_privileged_first_index, non_privileged_last_index]
    obs_dict['privileged']     = [privileged_first_index, privileged_last_index]

    return obs_dict