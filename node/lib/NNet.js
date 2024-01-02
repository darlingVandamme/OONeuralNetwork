import Neuron from "./neuron.js";

class NNet{
    constructor(inputSize){
        this.layers = 0
        this.step = 3.0
        this.batchSize = 10
        this.trainings = 0
        this.batches =0
        this.count = 0
        // keep a list of neurons
        // makes it easier to reset and serialize
        this.neurons = [] // Hidden and output Neurons
        this.allNeurons = []
        this.output = []
        this.input = this.addLayer(inputSize)
        this.costs = []
        this.costIndex=0
        this.costsSize = 500
    }

    addLayer(size){
        let position = 0
        let layer = Array.from({length:size}, (n1,i)=>{
            let n = new Neuron()
            n.position = position++
            n.id=this.allNeurons.length
            n.index = i
            this.allNeurons.push(n)
            this.output.forEach((other)=> n.connect(other))
            return n
        })
        this.neurons.push(...layer.filter(n=> !n.isInput())) // add all non input neurons
        this.output = layer
        this.layers++
        return layer
    }

    reset(){this.neurons.forEach((n)=>{n.reset()})}

    feed(input){
        this.count++
        this.reset()
        this.input.forEach((n,i)=> {n.value = input[i]})
        this.neurons.forEach((n, i)=>{n.ff()})
    }

    getOutput(){
        return this.output.map(n=>n.getValue())
    }

    // convenience methods
    translateInput(value){
        // allow translate of any object to list of values
        // default do nothing
        return value
    }

    translateOutput(value){
        // allow translate of list of values to object
        // default do nothing
        return value
    }

    translateExpected(value){
        // allow translate of object to expected output vector
        // default do nothing
        return value
    }

    check(item){
        this.feed(this.translateInput(item))
        return this.translateOutput(this.getOutput())
    }

    //getCost(result)
    getCost(){
        return this.output.reduce((previous,n,i)=> (previous + (((n.expected - n.value) ** 2  ) / (2*this.output.length))) , 0)
    }
    getAverageCost(length){
        // average cost of the most recent trainins
        let slice
        if(!length){
            slice = this.costs
        } else {
            slice = this.costs.slice(Math.max(0,this.costIndex-length), this.costIndex+1)
            if (this.costIndex<length) {
                slice = [...slice, ...this.costs.slice(this.costIndex - length)]
            }
        }
        return slice.reduce((previous,cost)=>{return previous+cost},0)/slice.length
    }

    getHighest(pattern){
        return pattern.reduce((previous,val,i,arr)=> val>arr[previous]?i:previous,0)
    }

    train(item, expected){
        let output = this.check(item)
        // store the expected values in the output neurons
        let expectedVector = this.translateExpected(expected)
        this.output.forEach((n,i)=> {n.expected = expectedVector[i] })
        this.costIndex = this.trainings % this.costsSize
        this.costs[this.costIndex] = this.getCost()
        this.trainings ++

        // backpropagate Delta recursive
        this.neurons.forEach((n)=>n.getDelta())

        // adjust weights
        if (this.trainings % this.batchSize == 0){
            this.batches++
            // iterative
            this.neurons.forEach((n)=>n.learn(this.step/this.batchSize))
        }
    }

    toJSON(){
        // recent cost functions or success rate
        // trainings
        // patterns?

        return {
            inputSize: this.input.length,
            outputSize: this.output.length,
            allNeurons: this.allNeurons.map(n => {
                return {
                    bias: n.bias,
                    id: n.id,
                    index: n.index,
                    connectionIDs: n.getConnectionIDs(),
                    weights: n.getWeights()
                }
            })
        }
    }
    /*printInfo(){
        console.log("counts "+counts)
    }*/

}


export default NNet