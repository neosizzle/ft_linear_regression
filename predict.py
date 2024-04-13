import csv

def read_csv() :
	b1 = 0
	b0 = 0
	try:
		file=open("values.csv", "r")
		reader = csv.reader(file)
		for line in reader:
			t=line
			b1 = float(t[0])
			b0 = float(t[1])
	except:
		print("error parsing values.csv , b1 and b0 are set to 0")

	return (b1, b0)

b1, b0 = read_csv()
print(f"b1 {b1}, b0 {b0}")
try :
	x = int(input("Enter mileage: "))
	price = b1 * x + b0
	print(f"Predicted price: {price}")
except :
	print("tralalala bad input")