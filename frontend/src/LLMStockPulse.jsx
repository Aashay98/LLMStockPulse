import React, { useState } from 'react'
import StockResponse from './StockResponse';
import { Button, TextField } from '@mui/material';

function LLMStockPulse() {
    const [stockInput, setStockInput] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');


    const submitQuestion = () => {
        
    }
    const handleStockInput = (event) => {
        setStockInput(event.target.value);
    }
    return (
        <div>
            <div>
                <TextField id="outlined-basic" 
                    label="About which stock whould you like to learn about?" 
                    variant="outlined" 
                    name="stock_input" 
                    value={stockInput} 
                    onChange={handleStockInput}
                    sx={{ mt: 2 }} 
                    margin="normal"/>
                <Button variant="contained"
                    sx={{ mt: 2 }}   
                    onClick={submitQuestion}>
                        Submit
                </Button>
                    </div>
            <StockResponse/>
        </div>
    )
}

export default LLMStockPulse