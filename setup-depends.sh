sudo apt update && sudo apt upgrade -y
mkdir -p downloads
cd downloads

curl -o jdk-17.0.19_linux-x64_bin.tar.gz https://countbox.blob.core.windows.net/owhwarehouse/stock/jdk-17.0.19_linux-x64_bin.tar.gz

sudo tar -xzf jdk-17.0.19_linux-x64_bin.tar.gz

mkdir -p /opt/
sudo cp -r jdk-17.0.19/ /opt

curl -o spark-4.0.2-bin-hadoop3.tgz https://dlcdn.apache.org/spark/spark-4.0.2/spark-4.0.2-bin-hadoop3.tgz

sudo tar -xzf spark-4.0.2-bin-hadoop3.tgz
sudo cp -r spark-4.0.2-bin-hadoop3/ /opt

echo 'JAVA_HOME=/opt/jdk-17.0.19' >> ~/.bashrc
echo 'SPARK_HOME=/opt/spark-4.0.2-bin-hadoop3' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
source ~/.bashrc

echo "Setup completed successfully. Please restart your terminal or run 'source ~/.bashrc' to apply the changes."