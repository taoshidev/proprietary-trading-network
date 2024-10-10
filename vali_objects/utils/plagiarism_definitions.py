
from vali_objects.utils.plagiarism_events import PlagiarismEvents
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np
from vali_config import ValiConfig
import bittensor as bt


THRESHOLD = 0.99 #This will be changed to different variables for each subplagiarism

class FollowPercentage(PlagiarismEvents):

  def __init__(self, plagiarist_id):
    super().__init__(plagiarist_id, "follow")
  

  # Every class overrides the score function that will sometimes take different arguments

  def score(self, plagiarist_trade_pair, victim_key):

    plagiarist_orders = PlagiarismEvents.positions[(self.plagiarist_id, plagiarist_trade_pair)]
    victim_orders = PlagiarismEvents.positions[victim_key]
    event_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    
    if event_key in self.time_differences:
      differences = self.time_differences[event_key]

    differences, copied_victim_orders, copy_plagiarist_orders = FollowPercentage.compute_time_differences(plagiarist_orders, victim_orders)
    self.time_differences[event_key] = differences
    percent_of_follow = len(differences)/len(victim_orders) if len(victim_orders) > 0 else 0

    #if percent_of_follow >= THRESHOLD:

    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    #"copied_victim_orders": copied_victim_orders, #TODO maybe keep something like this??
                                    #"copy_plagiarist_orders": copy_plagiarist_orders,
                                    "type": self.name,
                                    "score": percent_of_follow
                                    }
  

  def compute_time_differences(plagiarist_orders, victim_orders):

    time_resolution = ValiConfig.PLAGIARISM_MATCHING_TIME_RESOLUTION_MS
    time_window =  time_resolution * 60 * 12

    differences = []
    i = j = 0

    copied_victim_orders = []
    copy_plagiarist_orders = []
    while i < len(victim_orders) and j < len(plagiarist_orders):

        difference = plagiarist_orders[j]["start"] - (victim_orders[i]["start"])

        if difference <= time_window and difference >= 0:
            
            differences.append(difference / time_resolution)
            #copied_victim_orders.append(victim_orders[i])
            #copy_plagiarist_orders.append(plagiarist_orders[j])

            i += 1
            j = 0
        else:
            
            j+= 1

    return differences, copied_victim_orders, copy_plagiarist_orders


  
  def average_time_lag(plagiarist_orders, victim_orders, differences=None):
    if differences == None:
      differences, _, _ = FollowPercentage.compute_time_differences(plagiarist_orders, victim_orders)

    avg_difference = int(sum(differences)/len(differences)) if len(differences) > 0 else 0

    return avg_difference


class LagDetection(PlagiarismEvents):
  
  def __init__(self, plagiarist_id):
    super().__init__(plagiarist_id, "lag")

  
  def score(self, plagiarist_trade_pair, victim_key):

    plagiarist_score = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    victim_score = CopySimilarity.score_direct(victim_key[0], victim_key[1], self.plagiarist_id, plagiarist_trade_pair)

    lag_score = plagiarist_score / victim_score if victim_score > 0 else 0

    # Add environment variable here
    #if lag_score >= 1.005: #Too many nonPlagiarism events pass this, change the threshold

    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    "type": self.name,
                                    "score": lag_score
                                }
    
  
class CopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id):
    super().__init__(plagiarist_id, "single")


  def score(self, plagiarist_trade_pair, victim_key):

    similarity = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])
    #if similarity >= THRESHOLD:
    plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, victim_key[0], victim_key[1])

    self.metadata[plagiarism_key] = {"victim": victim_key[0],
                                    "victim_trade_pair": victim_key[1],
                                    "type": self.name,
                                    "score": similarity
                                    }
  

  def score_direct(plagiarist_id, plagiarist_trade_pair, victim_id, victim_trade_pair):

    plagiarist_vector = PlagiarismEvents.rasterized_positions[(plagiarist_id, plagiarist_trade_pair)]
    victim_vector = PlagiarismEvents.rasterized_positions[(victim_id, victim_trade_pair)]

    plagiarist_positions = PlagiarismEvents.positions[(plagiarist_id, plagiarist_trade_pair)]
    victim_positions = PlagiarismEvents.positions[(victim_id, victim_trade_pair)]

    differences = None
    event_key = (plagiarist_id, plagiarist_trade_pair, victim_id, victim_trade_pair)
    if event_key in PlagiarismEvents.time_differences:
      differences = PlagiarismEvents.time_differences[event_key]

    time_lag = FollowPercentage.average_time_lag(plagiarist_positions, victim_positions, differences=differences)
    if event_key in PlagiarismEvents.copy_similarities:
      similarity = PlagiarismEvents.copy_similarities[event_key]
    elif time_lag > 0:
      similarity = cosine_similarity([plagiarist_vector[:-time_lag]], [victim_vector[time_lag:]])[0][0]

    else:

      similarity = cosine_similarity([plagiarist_vector], [victim_vector])[0][0]

    PlagiarismEvents.copy_similarities[event_key] = similarity
    if similarity > .9:
      bt.logging.info(f"score: {similarity}")
      bt.logging.info(f"people: {(plagiarist_id, plagiarist_trade_pair, victim_id, victim_trade_pair)}")
    return similarity
  

# Add fields that show all victims involved in this combination plagiarism

class TwoCopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id):
    super().__init__(plagiarist_id, "two")


  def score_all(self, plagiarist_trade_pair):
    single_pair_similarities = {}
    for miner_id in PlagiarismEvents.miner_ids:
      for trade_pair in PlagiarismEvents.trade_pairs:
        event_key = (self.plagiarist_id, plagiarist_trade_pair, miner_id, trade_pair)
        if miner_id != self.plagiarist_id:
          victim_key = (miner_id, trade_pair)
          if event_key in self.copy_similarities:
            single_pair_similarities[victim_key] = self.copy_similarities[event_key]
          else: 
            single_pair_similarities[victim_key] = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, miner_id, trade_pair)

    self.score(plagiarist_trade_pair, single_pair_similarities)


  def score(self, plagiarist_trade_pair, single_pair_similarities):
    # Requires a single_pair_similarities to compute average of top scorers
    top_pairs = heapq.nlargest(2, single_pair_similarities.items(), key=lambda x: x[1])
    two_n_avg = np.average([x[1] for x in top_pairs])
    #if two_n_avg >= THRESHOLD:

    for pair in top_pairs:
      plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, pair[0][0], pair[0][1])
      self.metadata[plagiarism_key] = {"victim": pair[0][0],
                                      "victim_trade_pair": pair[0][1],
                                      "type": self.name,
                                      "score": two_n_avg
                                        }


# Add fields that show all victims involved in this combination plagiarism

class ThreeCopySimilarity(PlagiarismEvents):
  
  def __init__(self, plagiarist_id):
    super().__init__(plagiarist_id, "three")

  def score_all(self, plagiarist_trade_pair):

    single_pair_similarities = {}
    for miner_id in PlagiarismEvents.miner_ids:
      for trade_pair in PlagiarismEvents.trade_pairs:
        event_key = (self.plagiarist_id, plagiarist_trade_pair, miner_id, trade_pair)
        if miner_id != self.plagiarist_id:
          victim_key = (miner_id, trade_pair)
          if event_key in self.copy_similarities:
            single_pair_similarities[victim_key] = self.copy_similarities[event_key]
          else:
            single_pair_similarities[victim_key] = CopySimilarity.score_direct(self.plagiarist_id, plagiarist_trade_pair, miner_id, trade_pair)

    self.score(plagiarist_trade_pair, single_pair_similarities)


  def score(self, plagiarist_trade_pair, single_pair_similarities):

    top_pairs = heapq.nlargest(3, single_pair_similarities.items(), key=lambda x: x[1])
    three_n_avg = np.average([x[1] for x in top_pairs])
    #if three_n_avg >= THRESHOLD:
    for pair in top_pairs:
      plagiarism_key = (self.plagiarist_id, plagiarist_trade_pair, pair[0][0], pair[0][1])
      self.metadata[plagiarism_key] = {"victim": pair[0][0],
                                      "victim_trade_pair": pair[0][1],
                                      "type": self.name,
                                      "score": three_n_avg
                                        }