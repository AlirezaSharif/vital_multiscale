class BankAccount:
    """
    A class used to represent an bank account. This class is a template for the various usages associated with classes.
    Attributes
    ----------
    __MIN_BALANCE : int
        minimum allowed balance
    Methods
    -------
    deposit(amount=0)
        addas amount to balance
    withdraw(amount=0)
        subtracts amount from balance
    from_csv(filepath)
        returns class instance from csv file
    """
    
    __MIN_BALANCE = -10_000

    
    def __init__(self, owner, account_number, balance=0):
        self._owner = owner    
        self._account_number = account_number
        self._created_at = datetime.now().date()
        if balance < self.__MIN_BALANCE:
            raise ValueError("Balance to small!")
        else:
            self._balance = balance
        
    def __eq__(self, other):
        return True if self._account_number == other._account_number else False
    
    def __str__(self):
        return f"""
        Bank Account:
            Account Owner: {self._owner}
            Account Number: {self._account_number}
            Creation Date: {self._to_dash_date(str(self.created_at))}
            Current Balance: {self._balance}
        """
    
    def __repr__(self):
        return f"BankAccount(owner='{self._owner}', " \
               f"account_number={self._account_number}, " \
               f"balance={self._balance})"
    
    @classmethod
    def from_csv(cls, filepath):
        with open(filepath, "r") as f:
            row = csv.reader(f).__next__()
            owner, account_number = row
        return cls(owner, account_number)
    
    @staticmethod
    def _to_dash_date(date):
        # utility to replace "/" with "-" in a date
        return date.replace("/", "-") 
    
    @property
    def account_number(self):
        return self._account_number
    
    @property
    def created_at(self):
        return self._created_at
    
    @property
    def balance(self):
        return self._balance
    
    @balance.setter
    def balance(self, new_balance):
        if new_balance < self.__MIN_BALANCE:
            raise ValueError("Balance to small!")
        else:
            self._balance = new_balance
        
    def deposit(self, amount=0):
        self.balance += amount
        
    def withdraw(self, amount=0):
        self.balance -= amount