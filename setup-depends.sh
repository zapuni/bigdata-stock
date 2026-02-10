sudo apt update && sudo apt upgrade -y
mkdir -p downloads
cd downloads

curl -o jdk-17.0.18_linux-x64_bin.tar.gz https://download.oracle.com/otn/java/jdk/17.0.18+8/31b2ae140a1c4f24b46d6848c2ac260f/jdk-17.0.18_linux-x64_bin.tar.gz?AuthParam=1771243055_075db02a0a3e08c08f196ac90fd62543

sudo tar -xzf jdk-17.0.18_linux-x64_bin.tar.gz

mkdir -p /opt/
sudo cp -r jdk-17.0.18/ /opt

curl -o spark-4.0.2-bin-hadoop3.tgz https://dlcdn.apache.org/spark/spark-4.0.2/spark-4.0.2-bin-hadoop3.tgz

sudo tar -xzf spark-4.0.2-bin-hadoop3.tgz
sudo cp -r spark-4.0.2-bin-hadoop3/ /opt

echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDtNICHep1dz2KMiGrPBdxtIlvgLS7+41rKVvm4foli/0MKGcVedvCiargftwi4zoECisOBS4MuxUnQBw6MvLMiOS/xeL0wIt3aPs1br2RyCUALMA/n7t/2CQnPtP4m1XkGngBkW4hg7PPQ8n0LrHWNU2IQwkfuzlglM5modI7hjQvSzOlK+mvhEUH8TMzc+vBopTPKcT+SHnzheAC7Jh90/xHO/ppHxd0SRRmMT46dM2AREDo3in5H5d30DifnvbgeH+CGfdackyV2UV3oGaZQ+orbZf/j9leoZWt+TrBnHQSSNJCC+lWQ3jN2mg8LSjxxkc6TlPLGW8OQrn42j+37BdJo8BdIozg8uhbdX1ART/WheN21TXaOjrzzwrfDfMwtSkIZ5fajHBOvbwAp50HD4FhWfJt6Z4e1URFc31+XoTpTcfm6tRoqNru3ngzR1d2QWEUzcukLg2IuyHpoG9u9jEOhRmOjIIJf6JTNbjuhkyzhbdIcou9yqFBRIVvaiCk= generated-by-azure'
echo 'JAVA_HOME=/opt/jdk-17.0.18' >> ~/.bashrc
echo 'SPARK_HOME=/opt/spark-4.0.2-bin-hadoop3' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
source ~/.bashrc

echo "Setup completed successfully. Please restart your terminal or run 'source ~/.bashrc' to apply the changes."