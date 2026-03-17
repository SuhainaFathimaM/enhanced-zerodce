import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from zerodce_trainer import ZeroDCETrainer, create_dataloaders
from enhanced_zerodce import EnhancedZeroDCENet, EnhancedZeroDCEEnhancer

class HyperparameterOptimizer:
    """
    Hyperparameter Optimization for Zero-DCE Innovation
    
    This will help you find the best hyperparameters for your enhanced Zero-DCE
    """
    
    def __init__(self, data_dir, save_dir="hyperparameter_search"):
        self.data_dir = data_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(data_dir, batch_size=2)
        
        print(f"Hyperparameter optimizer initialized")
        print(f"Dataset: {data_dir}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def define_search_space(self):
        """
        Define hyperparameter search space
        """
        search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-7, 1e-6, 1e-5, 1e-4],
            'batch_size': [2, 4, 8],
            'scheduler_type': ['cosine', 'step', 'plateau'],
            'iteration_count': [6, 8, 12],
            'loss_weights': {
                'spatial': [1, 5, 10],
                'exposure': [5, 10, 20],
                'color': [1, 5, 10],
                'smoothness': [100, 200, 500],
                'perceptual': [0.05, 0.1, 0.2]
            }
        }
        
        return search_space
    
    def grid_search(self, max_trials=20):
        """
        Perform grid search for best hyperparameters
        """
        search_space = self.define_search_space()
        
        # Generate combinations (limited for practicality)
        lr_options = search_space['learning_rate'][:3]
        wd_options = search_space['weight_decay'][:2]
        scheduler_options = search_space['scheduler_type'][:2]
        iteration_options = search_space['iteration_count'][:2]
        
        combinations = list(itertools.product(lr_options, wd_options, scheduler_options, iteration_options))
        
        results = []
        
        print(f"Starting grid search with {len(combinations)} combinations...")
        
        for trial, (lr, wd, scheduler, iterations) in enumerate(combinations[:max_trials]):
            print(f"\n{'='*50}")
            print(f"Trial {trial+1}/{min(max_trials, len(combinations))}")
            print(f"LR: {lr}, WD: {wd}, Scheduler: {scheduler}, Iterations: {iterations}")
            print(f"{'='*50}")
            
            # Train with current hyperparameters
            result = self.train_with_hyperparameters(lr, wd, scheduler, iterations, epochs=20)
            
            result['hyperparameters'] = {
                'learning_rate': lr,
                'weight_decay': wd,
                'scheduler_type': scheduler,
                'iteration_count': iterations
            }
            
            results.append(result)
            
            # Save intermediate results
            self.save_results(results, f"intermediate_trial_{trial+1}.json")
            
            print(f"Trial {trial+1} completed. Best Val Loss: {result['best_val_loss']:.6f}")
        
        # Find best hyperparameters
        best_result = min(results, key=lambda x: x['best_val_loss'])
        
        print(f"\n{'='*60}")
        print("GRID SEARCH COMPLETED")
        print(f"{'='*60}")
        print(f"Best hyperparameters: {best_result['hyperparameters']}")
        print(f"Best validation loss: {best_result['best_val_loss']:.6f}")
        print(f"Training time: {best_result['training_time']:.2f} seconds")
        
        # Save final results
        self.save_results(results, "grid_search_results.json")
        
        return best_result, results
    
    def train_with_hyperparameters(self, lr, wd, scheduler_type, iterations, epochs=20):
        """
        Train model with specific hyperparameters
        """
        # Initialize model with specified iterations
        model = EnhancedZeroDCENet(iteration=iterations, use_attention=True, multi_scale=True)
        
        # Initialize trainer
        trainer = ZeroDCETrainer(model=model)
        trainer.setup_training(learning_rate=lr, weight_decay=wd, scheduler_type=scheduler_type)
        
        # Train model
        start_time = time.time()
        
        # Create temporary save directory
        temp_save_dir = self.save_dir / f"temp_trial_{lr}_{wd}_{scheduler_type}_{iterations}"
        temp_save_dir.mkdir(exist_ok=True)
        
        try:
            history = trainer.train(
                self.train_loader, 
                self.val_loader, 
                num_epochs=epochs,
                save_dir=str(temp_save_dir),
                save_interval=10,
                early_stopping_patience=10
            )
            
            training_time = time.time() - start_time
            
            # Get best validation loss
            best_val_loss = min([h['val_loss'] for h in history if h['val_loss'] is not None])
            
            result = {
                'best_val_loss': best_val_loss,
                'training_time': training_time,
                'epochs_trained': len(history),
                'final_lr': history[-1]['learning_rate'],
                'history': history
            }
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_save_dir, ignore_errors=True)
            
            return result
            
        except Exception as e:
            print(f"Training failed: {e}")
            return {
                'best_val_loss': float('inf'),
                'training_time': time.time() - start_time,
                'epochs_trained': 0,
                'error': str(e)
            }
    
    def save_results(self, results, filename):
        """
        Save search results
        """
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {filepath}")
    
    def analyze_results(self, results):
        """
        Analyze hyperparameter search results
        """
        print(f"\n{'='*50}")
        print("HYPERPARAMETER ANALYSIS")
        print(f"{'='*50}")
        
        # Filter successful trials
        successful_results = [r for r in results if r.get('best_val_loss', float('inf')) < float('inf')]
        
        if not successful_results:
            print("No successful trials found")
            return
        
        # Sort by validation loss
        successful_results.sort(key=lambda x: x['best_val_loss'])
        
        print(f"Total trials: {len(results)}")
        print(f"Successful trials: {len(successful_results)}")
        print(f"Best validation loss: {successful_results[0]['best_val_loss']:.6f}")
        print(f"Worst validation loss: {successful_results[-1]['best_val_loss']:.6f}")
        
        # Analyze hyperparameter impact
        lr_impact = {}
        wd_impact = {}
        scheduler_impact = {}
        iteration_impact = {}
        
        for result in successful_results:
            hp = result['hyperparameters']
            loss = result['best_val_loss']
            
            # Learning rate impact
            lr = hp['learning_rate']
            if lr not in lr_impact:
                lr_impact[lr] = []
            lr_impact[lr].append(loss)
            
            # Weight decay impact
            wd = hp['weight_decay']
            if wd not in wd_impact:
                wd_impact[wd] = []
            wd_impact[wd].append(loss)
            
            # Scheduler impact
            scheduler = hp['scheduler_type']
            if scheduler not in scheduler_impact:
                scheduler_impact[scheduler] = []
            scheduler_impact[scheduler].append(loss)
            
            # Iteration impact
            iterations = hp['iteration_count']
            if iterations not in iteration_impact:
                iteration_impact[iterations] = []
            iteration_impact[iterations].append(loss)
        
        print(f"\nLearning Rate Impact:")
        for lr, losses in lr_impact.items():
            print(f"  {lr}: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        
        print(f"\nWeight Decay Impact:")
        for wd, losses in wd_impact.items():
            print(f"  {wd}: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        
        print(f"\nScheduler Impact:")
        for scheduler, losses in scheduler_impact.items():
            print(f"  {scheduler}: {np.mean(losses):.6f} ± {np.std(losses):.6f}")
        
        print(f"\nIteration Count Impact:")
        for iterations, losses in iteration_impact.items():
            print(f"  {iterations}: {np.mean(losses):.6f} ± {np.std(losses):.6f}")


class ZeroDCEInnovationFramework:
    """
    Complete framework for Zero-DCE innovation
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run_innovation_pipeline(self):
        """
        Complete innovation pipeline
        """
        print("="*60)
        print("ZERO-DCE INNOVATION PIPELINE")
        print("="*60)
        
        # Step 1: Train baseline model
        print("\nStep 1: Training baseline Zero-DCE...")
        baseline_trainer = ZeroDCETrainer()
        baseline_trainer.setup_training(learning_rate=1e-4)
        
        train_loader, val_loader = create_dataloaders(self.data_dir, batch_size=4)
        baseline_history = baseline_trainer.train(
            train_loader, val_loader, 
            num_epochs=50,  # Reduced for demo
            save_dir="baseline_zerodce",
            early_stopping_patience=20
        )
        
        baseline_best_loss = min([h['val_loss'] for h in baseline_history if h['val_loss'] is not None])
        print(f"Baseline best validation loss: {baseline_best_loss:.6f}")
        
        # Step 2: Hyperparameter optimization
        print("\nStep 2: Hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(self.data_dir)
        best_hp, all_results = optimizer.grid_search(max_trials=10)
        optimizer.analyze_results(all_results)
        
        # Step 3: Train enhanced model with best hyperparameters
        print("\nStep 3: Training enhanced Zero-DCE with best hyperparameters...")
        enhanced_model = EnhancedZeroDCENet(
            iteration=best_hp['hyperparameters']['iteration_count'],
            use_attention=True,
            multi_scale=True
        )
        
        enhanced_trainer = ZeroDCETrainer(model=enhanced_model)
        enhanced_trainer.setup_training(
            learning_rate=best_hp['hyperparameters']['learning_rate'],
            weight_decay=best_hp['hyperparameters']['weight_decay'],
            scheduler_type=best_hp['hyperparameters']['scheduler_type']
        )
        
        enhanced_history = enhanced_trainer.train(
            train_loader, val_loader,
            num_epochs=50,
            save_dir="enhanced_zerodce",
            early_stopping_patience=20
        )
        
        enhanced_best_loss = min([h['val_loss'] for h in enhanced_history if h['val_loss'] is not None])
        print(f"Enhanced best validation loss: {enhanced_best_loss:.6f}")
        
        # Step 4: Compare results
        print("\nStep 4: Innovation results...")
        improvement = ((baseline_best_loss - enhanced_best_loss) / baseline_best_loss) * 100
        
        print(f"{'='*50}")
        print("INNOVATION RESULTS")
        print(f"{'='*50}")
        print(f"Baseline Zero-DCE:     {baseline_best_loss:.6f}")
        print(f"Enhanced Zero-DCE:     {enhanced_best_loss:.6f}")
        print(f"Improvement:           {improvement:.2f}%")
        
        if improvement > 0:
            print("🎉 INNOVATION SUCCESSFUL! Enhanced Zero-DCE performs better!")
        else:
            print("⚠️  Enhanced model needs further optimization")
        
        # Save innovation report
        innovation_report = {
            'baseline_loss': baseline_best_loss,
            'enhanced_loss': enhanced_best_loss,
            'improvement_percent': improvement,
            'best_hyperparameters': best_hp['hyperparameters'],
            'baseline_history': baseline_history,
            'enhanced_history': enhanced_history,
            'all_hyperparameter_results': all_results
        }
        
        with open("innovation_report.json", 'w') as f:
            json.dump(innovation_report, f, indent=2, default=str)
        
        print(f"\nInnovation report saved to: innovation_report.json")
        
        return innovation_report


def main():
    """
    Main innovation pipeline
    """
    data_dir = "/home/sfm01/Downloads/luma/Lumanet-main/archive/lol_dataset/our485"
    
    framework = ZeroDCEInnovationFramework(data_dir)
    report = framework.run_innovation_pipeline()
    
    return report


if __name__ == "__main__":
    report = main()
