using EfficientNetSeq2label
using Test
using Random

@testset "EfficientNetSeq2label.jl" begin
    
    @testset "Hyperparameter Generation" begin
        Random.seed!(42)
        
        # Test default ranges
        ranges = EfficientNetSeq2label.nucleotide_ranges()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        @test hp.batch_size == 32
        @test num_layers(hp) >= 3
        @test num_layers(hp) <= 5
        
        # Test simple ranges
        ranges_simple = EfficientNetSeq2label.nucleotide_ranges_simple()
        hp_simple = generate_random_hyperparameters(batch_size=64, ranges=ranges_simple)
        @test hp_simple.batch_size == 64
        @test num_layers(hp_simple) >= 2
        @test num_layers(hp_simple) <= 3
        
        # Test amino acid ranges
        ranges_aa = EfficientNetSeq2label.amino_acid_ranges()
        hp_aa = generate_random_hyperparameters(batch_size=32, ranges=ranges_aa)
        @test hp_aa.batch_size == 32
    end
    
    @testset "Model Creation" begin
        Random.seed!(42)
        
        ranges = EfficientNetSeq2label.nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        
        # Create model on CPU
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        @test model isa SeqCNN
        @test model.hp == hp
        @test size(model.output_weights, 1) == 10  # output_dim
    end
    
    @testset "Forward Pass" begin
        Random.seed!(42)
        
        ranges = EfficientNetSeq2label.nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        
        # Create input
        sequences = randn(Float32, 4, 41, 1, 8)
        
        # Test forward pass
        predictions = predict_from_sequences(model, sequences; training=false)
        @test size(predictions) == (10, 8)
        @test eltype(predictions) == Float32
    end
    
    @testset "Intermediate Features" begin
        Random.seed!(42)
        
        ranges = EfficientNetSeq2label.nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        
        sequences = randn(Float32, 4, 41, 1, 4)
        
        # Test extracting features at different layers
        code_pwm = compute_code_at_layer(model, sequences, 0; training=false)
        @test ndims(code_pwm) == 4
        
        code_l1 = compute_code_at_layer(model, sequences, 1; training=false)
        @test ndims(code_l1) == 4
    end
    
    @testset "Loss Functions" begin
        Random.seed!(42)
        
        predictions = randn(Float32, 10, 8)
        targets = randn(Float32, 10, 8)
        
        # Test Huber loss
        loss = huber_loss(predictions, targets)
        @test loss isa Float32
        @test loss >= 0
        
        # Test with NaN values (should be handled)
        targets_nan = copy(targets)
        targets_nan[1, 1] = NaN32
        loss_nan = huber_loss(predictions, targets_nan)
        @test !isnan(loss_nan)
        
        # Test masked MSE
        mask = trues(10, 8)
        mask[1, 1] = false
        loss_masked = masked_mse(predictions, targets, mask)
        @test loss_masked isa Float32
        @test loss_masked >= 0
    end
    
    @testset "Model CPU/GPU Conversion" begin
        Random.seed!(42)
        
        ranges = EfficientNetSeq2label.nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        
        # Test model2cpu (should work on already-CPU model)
        model_cpu = model2cpu(model)
        @test model_cpu isa SeqCNN
        @test model_cpu.hp.batch_size == model.hp.batch_size
        @test num_layers(model_cpu.hp) == num_layers(model.hp)
    end
    
    @testset "Final Nonlinearity" begin
        # Test default (identity)
        ranges_default = EfficientNetSeq2label.nucleotide_ranges()
        @test ranges_default.final_nonlinearity === identity
        
        # Test tanh variants
        ranges_tanh = EfficientNetSeq2label.nucleotide_ranges_tanh()
        @test ranges_tanh.final_nonlinearity === tanh
        
        ranges_simple_tanh = EfficientNetSeq2label.nucleotide_ranges_simple_tanh()
        @test ranges_simple_tanh.final_nonlinearity === tanh
        
        ranges_fixed_tanh = EfficientNetSeq2label.nucleotide_ranges_fixed_pool_stride_tanh()
        @test ranges_fixed_tanh.final_nonlinearity === tanh
        
        # Test amino acid variants
        ranges_aa_tanh = EfficientNetSeq2label.amino_acid_ranges_tanh()
        @test ranges_aa_tanh.final_nonlinearity === tanh
        
        ranges_aa_fixed_tanh = EfficientNetSeq2label.amino_acid_ranges_fixed_pool_stride_tanh()
        @test ranges_aa_fixed_tanh.final_nonlinearity === tanh
        
        # Test model creation with identity
        hp_identity = EfficientNetSeq2label.generate_random_hyperparameters(
            batch_size=32, 
            ranges=ranges_default
        )
        model_identity = EfficientNetSeq2label.SeqCNN(
            hp_identity, (4, 41), 10; 
            final_nonlinearity=ranges_default.final_nonlinearity,
            use_cuda=false,
            pwm_dropout_p=0.0f0
        )
        @test model_identity.final_nonlinearity === identity
        
        # Test model creation with tanh
        hp_tanh = EfficientNetSeq2label.generate_random_hyperparameters(
            batch_size=32, 
            ranges=ranges_tanh
        )
        model_tanh = EfficientNetSeq2label.SeqCNN(
            hp_tanh, (4, 41), 10; 
            final_nonlinearity=ranges_tanh.final_nonlinearity,
            use_cuda=false,
            pwm_dropout_p=0.0f0
        )
        @test model_tanh.final_nonlinearity === tanh
        
        # Test create_model uses ranges.final_nonlinearity
        Random.seed!(42)  # Use fixed seed to ensure valid architecture
        model_auto_tanh = EfficientNetSeq2label.create_model(
            (4, 41), 10, 32;
            ranges=ranges_tanh,
            use_cuda=false
        )
        @test model_auto_tanh.final_nonlinearity === tanh
    end
    
    @testset "README Examples" begin
        # Test all examples from README work correctly
        
        # Basic Model Creation and Inference
        Random.seed!(42)
        ranges = nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        sequences = randn(Float32, 4, 41, 1, 8)
        predictions = predict_from_sequences(model, sequences; training=false)
        @test size(predictions) == (10, 8)
        
        # Using Different Hyperparameter Ranges
        Random.seed!(123)
        nuc_ranges = nucleotide_ranges()
        hp_nuc = generate_random_hyperparameters(batch_size=64, ranges=nuc_ranges)
        model_nuc = SeqCNN(hp_nuc, (4, 100), 5; use_cuda=false, pwm_dropout_p=0.0f0)
        @test model_nuc isa SeqCNN
        
        aa_ranges = amino_acid_ranges()
        hp_aa = generate_random_hyperparameters(batch_size=32, ranges=aa_ranges)
        model_aa = SeqCNN(hp_aa, (20, 50), 3; use_cuda=false, pwm_dropout_p=0.0f0)
        @test model_aa isa SeqCNN
        
        # Computing Loss
        Random.seed!(42)
        ranges = nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        sequences = randn(Float32, 4, 41, 1, 16)
        targets = randn(Float32, 10, 16)
        predictions = predict_from_sequences(model, sequences; training=false)
        loss = huber_loss(predictions, targets)
        @test loss >= 0
        
        # Extracting Intermediate Features
        Random.seed!(42)
        ranges = nucleotide_ranges_simple()
        hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
        model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)
        sequences = randn(Float32, 4, 41, 1, 4)
        code_pwm = compute_code_at_layer(model, sequences, 0; training=false)
        code_l1 = compute_code_at_layer(model, sequences, 1; training=false)
        @test ndims(code_pwm) == 4
        @test ndims(code_l1) == 4
    end
end
