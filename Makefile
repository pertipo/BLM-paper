target:
	sudo apt-get install nlohmann-json3-dev
	g++ -o main main.cpp

clean:
	rm main
