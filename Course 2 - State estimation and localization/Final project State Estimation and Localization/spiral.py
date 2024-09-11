class CarType:
    def __init__(self,segment) -> None:
        self.model = "Honda Jazz"
        self.segment = segment

class RentalCar:
    def __init__(self,type:CarType,iD) -> None:
        self.iD = iD
        self.type = type
        self.rented = False
        self.renterID = -1
        self.pricePerDay = 0


class Renter:
    def __init__(self,name, age) -> None:
        self.name = name
        self.age = age

class RenterWithCars(Renter):
    def __init__(self,person:Renter, iD, carId) -> None:
        super().__init__(person.name,person.age)
        self.iD = iD
        self.carId = carId
        self.startDate = "2 sep 2024"
        self.duration = 3

class CRC:
    def __init__(self) -> None:
        self.cars = []
        self.rentersWithCars = []
        self.num_cars = 0

    def rentCar(renter:Renter, type:CarType, startDate, duration):
        # filter cars
        # get 1st car if available
        # assign ID to renter
        # assign car id to renter
        # assign start and end dates to renter
        # update renter ID and renter status on car
        pass

    def receiveCar(renter: RenterWithCars):
        # Match end date
        # Take money
        # Remove renter from list
        # update renter status on car
        
        pass

    def filterCars(self,type:CarType):
        filter_cars = self.cars.copy()
        if type.model != None:
            filter_cars = [k for k in filter_cars if k.type.model == type.model]
        if type.category != None:
            filter_cars = [k for k in filter_cars if k.type.model == type.model]
        return filter_cars
    
    def addCar(self,type:CarType):
        car = RentalCar(iD=self.num_cars+1,type=type)
        self.cars.append(car)
        self.num_cars = self.num_cars + 1






# Set up company
hertz = CRC()
car1type = CarType("hatchback")
hertz.addCar(car1type)
car2type = CarType("sedan")
hertz.addCar(car2type)
car3type = CarType("sedan")
hertz.addCar(car3type)
car4type = CarType("SUV")
hertz.addCar(car4type)

for t in range(150):

    print("0: exit")
    print("1: Renter rents a car")
    print("2: Renter returns a car")
    print("3: Add car")  
    
    usinp = input("Enter choice:")
    if usinp == 0:

        print("Exiting")
        break
    elif usinp == 1:
        renter = Renter("Tim","24")

