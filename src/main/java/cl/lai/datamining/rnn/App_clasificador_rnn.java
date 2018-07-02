package cl.lai.datamining.rnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.smartcardio.ATR;

import weka.associations.Apriori;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.LWL;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class App_clasificador_rnn {
	public static String file = "/Users/chunhaulai/Desktop/datamining-weka/src/resources/arriendo_dpto_categoria_numerica_5atributos_clasificacion.csv";

	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
		// Lectura de archivo para obtener los valores nominales

		Map<Integer, String> correspondencia = new HashMap<Integer, String>();

		String aux = reader.readLine();
		int totalRegistros = 52;

		String arrays[] = aux.split(";");

		Instances dptos = null;
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		// omitir el primer, segundo atributo y el último atributo, y considerar
		// los 4 atributos restantes:
		for (int i = 2; i < arrays.length - 1; i++) {
			attributes.add(new Attribute(arrays[i]));
		}
		// considerar el ultimo atributo como target
		ArrayList<String> clasesPreviamenteDefinida = new ArrayList<String>(3);
		clasesPreviamenteDefinida.add("BAJO");
		clasesPreviamenteDefinida.add("INTERMEDIO");
		clasesPreviamenteDefinida.add("ALTO");
		Attribute classAttribute = new Attribute("SECTOR", clasesPreviamenteDefinida);
		attributes.add(classAttribute);

		Instances isTrainingSet = new Instances("traning", attributes, totalRegistros);
		isTrainingSet.setClassIndex(classAttribute.index());

		int filas = 1;
		// Lectura de cada instancia
		while ((aux = reader.readLine()) != null) {
			arrays = aux.split(";");
			correspondencia.put(filas, arrays[0]);

			DenseInstance inst = new DenseInstance(5);
			for (int at = 0, i = 2; i < arrays.length - 1; i++, at++) {
				double valor = Double.parseDouble(arrays[i]);
				inst.setValue(attributes.get(at), valor);
			}
			inst.setValue(classAttribute, arrays[arrays.length - 1]);

			isTrainingSet.add(inst);
			filas++;

		}

		MultilayerPerceptron rnn = new MultilayerPerceptron();
		String[] options = new String[] { "-N", "500" };// hacer que se itera 100 veces
		rnn.setOptions(options);
		rnn.buildClassifier(isTrainingSet);
		DenseInstance inst = new DenseInstance(5);

		Attribute a1 = attributes.get(1);// ALMACENES PEQUENOS VENTA DE  ALIMENTOS
		Attribute a2 = attributes.get(2);// ESTABLECIMIENTOS DE ENSENANZA PRIMARIA Y SECUNDARIA PARA ADULTOS
		Attribute a3 = attributes.get(3);// MANTENIMIENTO Y REPARACION DE VEHICULOS AUTOMOTORES
		Attribute a4 = attributes.get(4);// PELUQUERIAS Y SALONES DE BELLEZA
		inst.setValue(a1, 1377);
		inst.setValue(a2,493);
		inst.setValue(a3,404);
		inst.setValue(a4,963);
					
		// asociar la instancia a un dataset
		Instances dataset = new Instances("probar", attributes, 1);
		dataset.setClassIndex(classAttribute.index());
		inst.setDataset(dataset);

		int indiceClasificado = (int) rnn.classifyInstance(inst);

		System.out.println("SECTOR clasificado: " + clasesPreviamenteDefinida.get(indiceClasificado));

	}
}
