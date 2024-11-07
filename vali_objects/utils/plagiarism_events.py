from vali_objects.utils.reporting_utils import ReportingUtils

class PlagiarismEvents():
  positons = {}
  rasterized_positions = {}
  time_differences = {}
  copy_similarities = {}
  miner_ids = []
  trade_pairs = []
    
  def __init__(self, plagiarist_id, name):
    self.metadata = {} # Dictionary of data for plagiarism events
    self.plagiarist_id = plagiarist_id
    self.name = name


  def score_all(self, plagiarist_trade_pair):

    for miner_id in PlagiarismEvents.miner_ids:

      if miner_id != self.plagiarist_id:
        
        victim_key = (miner_id, plagiarist_trade_pair)
        self.score(plagiarist_trade_pair, victim_key)
    return 


  def score(self):
    return


  def summary(self) -> dict:
    return self.metadata


  def set_positions(positions, miner_ids, trade_pairs, current_time=None, lookback_window=None, time_resolution=None):

    PlagiarismEvents.positions = positions
    PlagiarismEvents.miner_ids = miner_ids
    PlagiarismEvents.trade_pairs = trade_pairs
    PlagiarismEvents.rasterized_positions = ReportingUtils.rasterize(positions, current_time=current_time, lookback_window=lookback_window, time_resolution=time_resolution)

  
  
  def clear_plagiarism_events():
    PlagiarismEvents.positons = {}
    PlagiarismEvents.rasterized_positions = {}
    PlagiarismEvents.time_differences = {}
    PlagiarismEvents.copy_similarities = {}
    PlagiarismEvents.miner_ids = []
    PlagiarismEvents.trade_pairs = []

