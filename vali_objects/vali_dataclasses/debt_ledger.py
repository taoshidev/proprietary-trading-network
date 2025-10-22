from penalty_ledger import PenaltyLedger
from emissions_ledger import EmissionsLedgerManager
from perf_ledger import PerfLedgerManager
"""
TODO:Make a DebtLedger dataclass to represent pnl data from PerfLedgers, all data in EmissionLedger, and all data in PenaltyLedger.
We will eventually add DebtLedgerManager class which handles updates, serialization, deserialization, and other operations on DebtLedger dataclass.
For now, it is more important to create the data structure so I can share with my team so they can start planning how to use the data in the UI.
"""