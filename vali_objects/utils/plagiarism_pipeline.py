from vali_objects.utils.plagiarism_definitions import FollowPercentage, LagDetection, CopySimilarity, TwoCopySimilarity, ThreeCopySimilarity
from vali_objects.utils.plagiarism_events import PlagiarismEvents
from vali_objects.utils.reporting_utils import ReportingUtils
from vali_objects.utils.position_utils import PositionUtils
from vali_config import ValiConfig
import uuid
import time


class PlagiarismPipeline:

  rasterized_positions = {} # ((miner_id, trade_pair): rasterized position)
  order_lists = {} # ((miner_id, trade_pair), all orders)
  current_time = 0
  
  def __init__(self, plagiarism_classes):
    self.overall_score = 0
    self.max_victim = None
    self.max_trade_pair = None

    self.plagiarism_classes = plagiarism_classes
    

  def generate_plagiarism_events(self, miner_ids, trade_pairs, plagiarist_id, state_dict) -> list[dict]:

    for trade_pair in trade_pairs:
        for sub_plagiarism in self.plagiarism_classes:

          sub_plagiarism.score_all(trade_pair)

    return self.compose(miner_ids, trade_pairs, plagiarist_id, state_dict)
  

  def compose_sub_plagiarism(self, metadatas, plagiarism_key) -> list[dict]:
    events = []
    # Have a list of the relevant plagiarism events 
    for sub_plagiarism in metadatas:
      
      if plagiarism_key in sub_plagiarism:

        event = sub_plagiarism[plagiarism_key]
        events.append({"type": event["type"],
                      "score": event["score"]})
        
    return events
     

  def compose_victims(self, miner_ids, trade_pairs, plagiarist_id, plagiarist_trade_pair, state_dict) -> list[dict]:

    metadatas = [x.summary() for x in self.plagiarism_classes]
    victims = []

    for miner_id in miner_ids:
      if miner_id == plagiarist_id:
        continue
      for trade_pair in trade_pairs:

        plagiarism_key = (plagiarist_id, plagiarist_trade_pair, miner_id, trade_pair)
        events = self.compose_sub_plagiarism(metadatas, plagiarism_key)
        lagPassed = followPassed = False
        current_score = 0

        for event in events:
          if event["type"] == "lag" and event["score"] >= ValiConfig.PLAGIARISM_FOLLOWER_TIMELAG_THRESHOLD:
            lagPassed = True
          elif event["type"] == "follow" and event["score"] >= ValiConfig.PLAGIARISM_FOLLOWER_SIMILARITY_THRESHOLD:
            followPassed = True
          if event["type"] == "single" or event["type"] == "two" or event["type"] == "three":
            current_score = max(current_score, event["score"])

        if current_score >= self.overall_score and (lagPassed or followPassed):
          self.overall_score = current_score
          self.max_trade_pair = plagiarist_trade_pair
          self.max_victim = {"victim": miner_id,
                    "victim_trade_pair": trade_pair,
                    "events": events}
        
        # If the plagiarist passes lag and follow thresholds and passes threshold for at least one other type of plagiarism, report
        # For the time being lag is only used as additional evidence for plagiarism rather than a necessary condition
        if len(events) >= 2 and followPassed and current_score >= ValiConfig.PLAGIARISM_REPORTING_THRESHOLD:

          positions = PlagiarismEvents.positions[(miner_id, trade_pair)]
          PlagiarismPipeline.rasterized_positions[(miner_id, trade_pair)] = ReportingUtils.rasterize_cumulative_position(positions, current_time=PlagiarismPipeline.current_time)
          PlagiarismPipeline.order_lists[(miner_id, trade_pair)] = state_dict[(miner_id, trade_pair)]
          if (plagiarist_id, plagiarist_trade_pair) not in PlagiarismPipeline.rasterized_positions:
            PlagiarismPipeline.rasterized_positions[(plagiarist_id, plagiarist_trade_pair)] = ReportingUtils.rasterize_cumulative_position(PlagiarismEvents.positions[(plagiarist_id, plagiarist_trade_pair)], current_time=PlagiarismPipeline.current_time)
            PlagiarismPipeline.order_lists[(plagiarist_id, plagiarist_trade_pair)] = PlagiarismEvents.positions[(plagiarist_id, plagiarist_trade_pair)]
        
          victim = {"victim": miner_id,
                    "victim_trade_pair": trade_pair,
                    "events": events}
          
          victims.append(victim)
        

    return victims


  def compose(self, miner_ids, trade_pairs, plagiarist_id, state_dict) -> list[dict]:

    plagiarism_data = {}

    for plagiarist_trade_pair in trade_pairs:
        
        victims = self.compose_victims(miner_ids, trade_pairs, plagiarist_id, plagiarist_trade_pair, state_dict)

        if len(victims) > 0:

          plagiarism_report = {"plagiarist_trade_pair": plagiarist_trade_pair,
                             "victims": victims}
        
          plagiarism_data[plagiarist_trade_pair] = plagiarism_report

    return plagiarism_data
                     
  def state_list_to_dict(miners, trade_pairs, state_list):

    state_dict = {}
    for miner in miners:
      for trade_pair in trade_pairs:
        state_dict[(miner, trade_pair)] = []

    for order in state_list:
      state_dict[(order["miner_id"], order["trade_pair"])].append(order)

    return state_dict
  
  def reformat_raster(self):
    new_raster = {}
    for key, value in self.rasterized_positions.items():
        if key[0] not in new_raster:
            new_raster[key[0]] = {}
        new_raster[key[0]][key[1]] = value.tolist()
    new_raster["created_timestamp_ms"] = int(time.time() * 1000)
    return new_raster

  def reformat_positions(self):
    new_positions = {}

    for key, value in self.order_lists.items():
        if key[0] not in new_positions:
            new_positions[key[0]] = {}
        new_positions[key[0]][key[1]] = value
    new_positions["created_timestamp_ms"] = int(time.time() * 1000)
    return new_positions


  def run_reporting(positions, current_time) -> tuple[dict, dict, dict]:
    flattened_positions = PositionUtils.flatten(positions)
    positions_list_translated = PositionUtils.translate_current_leverage(flattened_positions)
    miners, trade_pairs, state_list = PositionUtils.to_state_list(positions_list_translated, current_time=current_time)
    state_dict = PlagiarismPipeline.state_list_to_dict(miners, trade_pairs, state_list)
    PlagiarismPipeline.current_time = current_time

    PlagiarismEvents.set_positions(state_dict, miners, trade_pairs, current_time=current_time)
    rasterized_positions = {}
    positions_data = {}
    plagiarists_data = []

    #In the case of no miners
    pipeline = None
    for miner_id in miners:

      plagiarism_classes = [FollowPercentage(miner_id),
                            LagDetection(miner_id),
                            CopySimilarity(miner_id),
                            TwoCopySimilarity(miner_id),
                            ThreeCopySimilarity(miner_id)]
      
      pipeline = PlagiarismPipeline(plagiarism_classes)
      trade_pair_output = pipeline.generate_plagiarism_events(miners, trade_pairs, miner_id, state_dict)
      # If nothing above the thresholds, maintain info on the maximum
      if len(trade_pair_output) == 0 and pipeline.overall_score != 0:
      
        trade_pair_output[pipeline.max_trade_pair] = [{"plagiarist_trade_pair": pipeline.max_trade_pair,
                            "victims": [pipeline.max_victim]}]
        victim_id = pipeline.max_victim["victim"]
        victim_tp = pipeline.max_victim["victim_trade_pair"]

        positions = PlagiarismEvents.positions[(victim_id, victim_tp)]
        raster_vector = ReportingUtils.rasterize_cumulative_position(positions, current_time=current_time)

        pipeline.rasterized_positions[(victim_id, victim_tp)] = raster_vector
        pipeline.order_lists[(victim_id, victim_tp)] = positions

        pipeline.rasterized_positions[(miner_id, pipeline.max_trade_pair)] = ReportingUtils.rasterize_cumulative_position(PlagiarismEvents.positions[(miner_id, pipeline.max_trade_pair)], current_time=current_time)
        pipeline.order_lists[(miner_id, pipeline.max_trade_pair)] = PlagiarismEvents.positions[(miner_id, pipeline.max_trade_pair)]

      
      final_output = {"event_id": str(uuid.uuid4()),
                      "time": round(time.time() * 1000),
                      "plagiarist": miner_id,
                      "overall_score": pipeline.overall_score,
                      "trade_pairs": trade_pair_output
                    }
      

      rasterized_positions.update(pipeline.rasterized_positions)
      positions_data.update(pipeline.order_lists)
      plagiarists_data.append(final_output)
      
    if pipeline is not None:
      rasterized_positions = pipeline.reformat_raster()
      positions_data = pipeline.reformat_positions()
    return (plagiarists_data, rasterized_positions, positions_data)