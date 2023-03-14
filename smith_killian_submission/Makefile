
get-data:
	@mkdir -p project/data
	curl https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip > \
		project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip
	unzip project/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip -d project/data/


docker-volume:
	docker volume create --name=project


build:
	docker-compose build


start:
	docker-compose up -d


stop:
	docker-compose down


status:
	docker ps
