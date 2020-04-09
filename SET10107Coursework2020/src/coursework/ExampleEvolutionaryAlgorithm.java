package coursework;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;


import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */
			

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = tournamentSelection(); 
			Individual parent2 = tournamentSelection();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = WholeArithmetic(parent1, parent2);			
			
			//mutate the offspring
			creepMutate(children);
			//bitSwap(children);
			//scrambleMutation(children);
			
			//Sawtooth diversifier
			//sawTooth();
			
			// Evaluate the children
			evaluateIndividuals(children);	
		
			
			// Replace children in population
			replace(children);
			

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}
	
	private Individual getBestArray(ArrayList<Individual> aPopulation) {
		double bestFitness = Double.MAX_VALUE;
		Individual best = null;
		for(Individual individual : aPopulation){
			if(individual.fitness < bestFitness || best == null){
				best = individual;
				bestFitness = best.fitness;
			}
		}
		return best;
	}


	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual tournamentSelection() {
		ArrayList<Individual> candidates = new ArrayList<Individual>();
		
		for(int i =0; i < Parameters.tournamentSize; i++) {
			candidates.add(population.get(Parameters.random.nextInt(population.size())));
		}
		
		return getBestArray(candidates).copy();
	}
	
	 private Individual rouletteWheelSelection(){
	        Individual parent = new Individual();
	        double totalFitness = 0;
	        
	        for(int i = 0; i < population.size(); i++){
	            totalFitness += population.get(i).fitness;
	        }
	        double random = ThreadLocalRandom.current().nextDouble(0, totalFitness);
	        double sum = 0;
	        int i = 0;
	        while(sum <= random){ 
	            sum += population.get(i).fitness;
	            i++;
	        }
	        parent = population.get(i-1);
	        
	        return parent;
	    }

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	
	
	private ArrayList<Individual> onePointCrossover(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<Individual>();
		
		if(Parameters.random.nextDouble() > Parameters.crossoverProbability) {
			children.add(parent1.copy());
			children.add(parent2.copy());
		}
		else {
			int counter = 0;
			Individual child = new Individual();
			int crossoverPoint = Parameters.random.nextInt(parent1.chromosome.length);
			
			while(counter < 2)
			{
				for(int i=0; i < crossoverPoint; i++) {
					child.chromosome[i] = parent1.chromosome[i];
				}
				
				for(int i=crossoverPoint; i < parent2.chromosome.length; i++ )
				{
					child.chromosome[i] = parent2.chromosome[i];
				}
				
				children.add(child);
			}
		}
		
		
		
		return children;
		
	}
	
	private ArrayList<Individual> unifromCrossover(Individual parent1, Individual parent2){
		
		ArrayList<Individual> children = new ArrayList<Individual>();
		
		if(Parameters.random.nextDouble() > Parameters.crossoverProbability) {
			children.add(parent1.copy());
			children.add(parent2.copy());
		}
		
		int counter = 0;
		Individual child = new Individual();
		
		while(counter < 2) {
			for(int i =0; i < parent1.chromosome.length; i++) {
				if(Parameters.random.nextInt(2) == 0) {
					child.chromosome[i] = parent1.chromosome[i];
				}
				else
				{
					child.chromosome[i] = parent2.chromosome[i];
				}
				
				children.add(child);
			}
			
		}
		return children;
	}
	
	
	
	
	private ArrayList<Individual> WholeArithmetic(Individual parent1, Individual parent2) {
        ArrayList<Individual> children = new ArrayList<>();
        Individual child = new Individual();

        for (int i = 0; i < parent1.chromosome.length; i++) {
            child.chromosome[i] = ((parent1.chromosome[i] + parent2.chromosome[i])/2);
        }

        children.add(child);

        return children;
    }
	/**
	 * Mutation
	 * 
	 * 
	 */
	
	private void bitSwap(ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            for (int i = 0; i < individual.chromosome.length; i++) {
                if (Parameters.random.nextDouble() < Parameters.mutateRate) {
                    int crossoverPoint = Parameters.random.nextInt(individual.chromosome.length);
                    double temp = individual.chromosome[i];
                    individual.chromosome[i] = individual.chromosome[crossoverPoint];
                    individual.chromosome[crossoverPoint] = temp;
                }
            }
        }
    }
	
	private void creepMutate(ArrayList<Individual> individuals) {
        double mutationRate = Parameters.mutateRate;
        for(Individual individual : individuals) {
            for(int i = 0; i < mutationRate; i++){
            	if (Parameters.random.nextDouble() < Parameters.mutateRate) {
            		 int index = Parameters.random.nextInt(individual.chromosome.length);
                     individual.chromosome[index] += ThreadLocalRandom.current().nextDouble(-5, 5);
            	}
               
            }
        }
    }
	
	private void scrambleMutation(ArrayList<Individual> individuals) {
		 double mutationRate = Parameters.mutateRate;
		 for(Individual individual : individuals) {
			 if (Parameters.random.nextDouble() < Parameters.mutateRate) {
				 for(int i = 0; i < individuals.size(); i++) {
					 int point1 = randomNumber(0, individual.chromosome.length);
					 int point2 = randomNumber(0,individual.chromosome.length);
					 
					 while(point1 >= point2) {
						 point1 = randomNumber(0, individual.chromosome.length);
						 point2 = randomNumber(0,individual.chromosome.length);
					 }
					 
					 for(int j=0; j<5; j++) {
						 int index1 = randomNumber(point1, point2+1);
						 int index2 = randomNumber(point1, point2+1);
						 
						 double beforeScramble = individual.chromosome[index1];
						 individual.chromosome[index1] = individual.chromosome[index2];
						 individual.chromosome[index2] = beforeScramble;	 
					 }
				 }
			 }
			
			 
		 }
	}
	
	private void sawTooth(){
        if(evaluations % 100 == 0){
            if(population.size() >= 10){
            	int worst = getWorstIndex();
                population.remove(worst);
            }else{
                initialise();
            }
        }
    }

 

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			int idx = getWorstIndex();	
			population.remove(idx);
			population.add(individual);
		}		
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	
	private int randomNumber(int min , int max) {
		Random r = new Random();
		double d = min + r.nextDouble() * (max - min);
		return (int)d;
	}

	@Override
	public double activationFunction(double x) {
		/*
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
		*/
		return x;
	}
	
}
